use anyhow::{Context as _, Result, anyhow};
use futures::{Stream, StreamExt as _, stream};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use vllm_engine_core_client::mock_engine::MockEngineDataSockets;
use vllm_engine_core_client::protocol::utility::EngineCoreUtilityRequest;
use vllm_engine_core_client::protocol::{
    EngineCoreRequest, EngineCoreRequestBatchHeader, EngineCoreRequestType,
    decode_engine_core_requests, decode_msgpack, encode_msgpack,
};
use zeromq::{DealerSocket, PushSocket, SocketRecv as _, SocketSend as _, ZmqMessage};

use crate::engine::{EngineInput, EngineOutput};

/// Send one engine output batch to the client over the appropriate push socket.
async fn send_engine_outputs_to_client(
    push_sockets: &mut [PushSocket],
    EngineOutput {
        client_index,
        outputs,
    }: EngineOutput,
) -> Result<()> {
    let message = ZmqMessage::from(encode_msgpack(&outputs)?);
    let push_socket = push_sockets
        .get_mut(client_index as usize)
        .ok_or_else(|| anyhow!("client index {client_index} is outside the configured range"))?;
    push_socket.send(message).await?;
    Ok(())
}

/// Create a stream of `EngineInput` by continuously receiving messages from the given dealer socket
/// and decoding them into `EngineInput`.
fn dealer_input_stream(
    dealer: DealerSocket,
    client_count: usize,
) -> impl Stream<Item = Result<EngineInput>> {
    stream::unfold(dealer, move |mut dealer| async move {
        let input = loop {
            let message =
                match dealer.recv().await.context("failed to receive message from dealer socket") {
                    Ok(message) => message,
                    Err(err) => break Err(err),
                };

            match decode_request(message, client_count) {
                Ok(input) => break Ok(input),
                Err(RequestDecodeError::Recoverable(err)) => {
                    warn!(%err, "failed to decode engine request message; ignoring");
                }
                Err(RequestDecodeError::Fatal(err)) => break Err(err),
            }
        };

        Some((input, dealer))
    })
}

#[derive(Debug)]
enum RequestDecodeError {
    Recoverable(anyhow::Error),
    Fatal(anyhow::Error),
}

/// Decode a `ZmqMessage` into an `EngineInput`.
fn decode_request(
    message: ZmqMessage,
    client_count: usize,
) -> std::result::Result<EngineInput, RequestDecodeError> {
    let frames = message.into_vec();
    if frames.is_empty() {
        return Err(RequestDecodeError::Recoverable(anyhow!(
            "empty engine request message"
        )));
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = EngineCoreRequestType::from_frame(request_type_frame) else {
        return Err(RequestDecodeError::Recoverable(anyhow!(
            "unknown engine request type: {:?}",
            request_type_frame
        )));
    };

    if request_type != EngineCoreRequestType::AddBatch && frames.len() != 2 {
        return Err(RequestDecodeError::Recoverable(anyhow!(
            "invalid frame count for engine request: {}",
            frames.len()
        )));
    }

    let input = match request_type {
        EngineCoreRequestType::Add => {
            let request: Box<EngineCoreRequest> = decode_msgpack(frames[1].as_ref())
                .map_err(|error| RequestDecodeError::Recoverable(error.into()))?;
            EngineInput::Request(request)
        }
        EngineCoreRequestType::AddBatch => {
            let header_frame = frames.get(1).ok_or_else(|| {
                RequestDecodeError::Fatal(anyhow!(
                    "ADD_BATCH message is missing its correlation header"
                ))
            })?;
            let header: EngineCoreRequestBatchHeader = decode_msgpack(header_frame.as_ref())
                .map_err(|error| {
                    RequestDecodeError::Fatal(anyhow!(
                        "failed to decode ADD_BATCH correlation header: {error}"
                    ))
                })?;
            if header.client_index as usize >= client_count {
                return Err(RequestDecodeError::Fatal(anyhow!(
                    "ADD_BATCH client index {} is outside the configured range",
                    header.client_index
                )));
            }

            match decode_engine_core_requests(&frames[2..]) {
                Ok(requests) => EngineInput::RequestBatch { header, requests },
                Err(error) => EngineInput::RequestBatchDecodeFailed {
                    header,
                    failure_message: format!("failed to decode request batch payload: {error}"),
                },
            }
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())
                .map_err(|error| RequestDecodeError::Recoverable(error.into()))?;
            EngineInput::Abort(request_ids)
        }
        EngineCoreRequestType::Utility => {
            let request: EngineCoreUtilityRequest = decode_msgpack(frames[1].as_ref())
                .map_err(|error| RequestDecodeError::Recoverable(error.into()))?;
            EngineInput::Utility(request)
        }
        EngineCoreRequestType::StartDpWave => EngineInput::StartDpWave,
    };

    Ok(input)
}

/// Run the main IO loop for the mock engine, continuously receiving and decoding raw messages from
/// the dealer sockets, sending them to the engine loop task via `input_tx`, and receiving
/// `EngineOutput` from the engine loop task via `output_rx` and sending them to the client over the
/// appropriate push socket, until `shutdown` is cancelled.
pub(crate) async fn run_io_loop(
    data_sockets: Vec<MockEngineDataSockets>,
    input_tx: mpsc::UnboundedSender<EngineInput>,
    mut output_rx: mpsc::Receiver<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let (dealers, mut push_sockets): (Vec<_>, Vec<_>) =
        data_sockets.into_iter().map(|sockets| (sockets.dealer, sockets.push)).unzip();
    let client_count = push_sockets.len();
    let mut input_streams = stream::select_all(
        dealers
            .into_iter()
            .map(|dealer| dealer_input_stream(dealer, client_count))
            .map(Box::pin),
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),

            output = output_rx.recv() => {
                let output = output
                    .ok_or_else(|| anyhow!("mock engine output channel closed"))?;
                send_engine_outputs_to_client(&mut push_sockets, output).await?;
            }

            input = input_streams.next() => {
                let input = input
                    .ok_or_else(|| anyhow!("mock engine input streams closed"))??;
                input_tx
                    .send(input)
                    .map_err(|_| anyhow!("mock engine state task shut down"))?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core_client::protocol::EngineCoreSamplingParams;

    fn header() -> EngineCoreRequestBatchHeader {
        EngineCoreRequestBatchHeader {
            client_index: 7,
            call_id: 42_u64.into(),
        }
    }

    fn requests() -> Vec<EngineCoreRequest> {
        vec![
            EngineCoreRequest {
                request_id: "req-1".to_string(),
                sampling_params: Some(EngineCoreSamplingParams::for_test()),
                client_index: 7,
                ..EngineCoreRequest::default()
            },
            EngineCoreRequest {
                request_id: "req-2".to_string(),
                sampling_params: Some(EngineCoreSamplingParams::for_test()),
                client_index: 7,
                ..EngineCoreRequest::default()
            },
        ]
    }

    #[test]
    fn add_batch_frame_decodes_as_one_batch_input() {
        let header = header();
        let requests = requests();
        let message = ZmqMessage::try_from(vec![
            EngineCoreRequestType::AddBatch.to_frame(),
            encode_msgpack(&header).unwrap().into(),
            encode_msgpack(&requests).unwrap().into(),
        ])
        .unwrap();

        let input = decode_request(message, 8).unwrap();
        let EngineInput::RequestBatch {
            header: decoded_header,
            requests: decoded_requests,
        } = input
        else {
            panic!("ADD_BATCH must remain one batch input");
        };

        assert_eq!(decoded_header, header);
        assert_eq!(decoded_requests, requests);
    }

    #[test]
    fn add_batch_payload_failure_preserves_correlation_header() {
        let header = header();
        let message = ZmqMessage::try_from(vec![
            EngineCoreRequestType::AddBatch.to_frame(),
            encode_msgpack(&header).unwrap().into(),
            vec![0xc1].into(),
        ])
        .unwrap();

        let input = decode_request(message, 8).unwrap();
        let EngineInput::RequestBatchDecodeFailed {
            header: decoded_header,
            failure_message,
        } = input
        else {
            panic!("malformed payload must become a correlated batch failure");
        };

        assert_eq!(decoded_header, header);
        assert!(failure_message.contains("failed to decode request batch payload"));
    }

    #[test]
    fn add_batch_header_failure_is_fatal() {
        let message = ZmqMessage::try_from(vec![
            EngineCoreRequestType::AddBatch.to_frame(),
            vec![0xc1].into(),
            encode_msgpack(&requests()).unwrap().into(),
        ])
        .unwrap();

        let Err(RequestDecodeError::Fatal(error)) = decode_request(message, 8) else {
            panic!("malformed correlation header must be fatal");
        };
        assert!(format!("{error:#}").contains("correlation header"));
    }

    #[test]
    fn add_batch_client_index_outside_configured_range_is_fatal() {
        let message = ZmqMessage::try_from(vec![
            EngineCoreRequestType::AddBatch.to_frame(),
            encode_msgpack(&header()).unwrap().into(),
            encode_msgpack(&requests()).unwrap().into(),
        ])
        .unwrap();

        let Err(RequestDecodeError::Fatal(error)) = decode_request(message, 7) else {
            panic!("out-of-range correlation client index must be fatal");
        };
        assert!(format!("{error:#}").contains("outside the configured range"));
    }
}
