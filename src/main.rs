use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    cf_account_id: String,
    cf_api_token: String,
    client: reqwest::Client,
}

// Hindsight 发来的 Cohere 格式请求
#[derive(Debug, Deserialize)]
struct CohereRerankRequest {
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
}

// 转发给 Cloudflare 的请求体
#[derive(Debug, Serialize)]
struct CloudflareContext {
    text: String,
}

#[derive(Debug, Serialize)]
struct CloudflareRerankRequest {
    query: String,
    // Cloudflare @cf/baai/bge-reranker-base 要求 contexts 是
    // [{text: "..."}, ...] 形式的数组，而不是字符串数组；字符串数组
    // 会触发 Cloudflare 端 code:8001 AiError: Invalid input。
    contexts: Vec<CloudflareContext>,
    top_k: usize,
}

// Cloudflare 返回的原始结构
#[derive(Debug, Deserialize)]
struct CloudflareResponse {
    result: CloudflareResult,
    success: bool,
    errors: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CloudflareResult {
    response: Vec<CloudflareItem>,
}

#[derive(Debug, Deserialize)]
struct CloudflareItem {
    id: usize,
    score: f64,
}

// 返回给 Hindsight 的 Cohere 格式
#[derive(Debug, Serialize)]
struct CohereRerankResponse {
    results: Vec<CohereResultItem>,
}

#[derive(Debug, Serialize)]
struct CohereResultItem {
    index: usize,
    relevance_score: f64,
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CohereRerankRequest>,
) -> impl IntoResponse {
    let doc_count = payload.documents.len();
    let top_n = payload.top_n.unwrap_or(doc_count).min(doc_count);

    let cf_payload = CloudflareRerankRequest {
        query: payload.query,
        contexts: payload
            .documents
            .into_iter()
            .map(|text| CloudflareContext { text })
            .collect(),
        top_k: top_n,
    };

    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/@cf/baai/bge-reranker-base",
        state.cf_account_id
    );

    let resp = state
        .client
        .post(&url)
        .header("Authorization", format!("Bearer {}", state.cf_api_token))
        .json(&cf_payload)
        .send()
        .await;

    match resp {
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();

            if !status.is_success() {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "error": format!("Cloudflare API error: {}", text)
                    })),
                )
                    .into_response();
            }

            match serde_json::from_str::<CloudflareResponse>(&text) {
                Ok(cf_data) => {
                    if !cf_data.success {
                        return (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({ "error": cf_data.errors })),
                        )
                            .into_response();
                    }

                    let mut results: Vec<CohereResultItem> = cf_data
                        .result
                        .response
                        .into_iter()
                        .map(|item| CohereResultItem {
                            index: item.id,
                            relevance_score: item.score,
                        })
                        .collect();

                    // Cohere 期望按相关性降序
                    results.sort_by(|a, b| {
                        b.relevance_score
                            .partial_cmp(&a.relevance_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    (StatusCode::OK, Json(CohereRerankResponse { results })).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Failed to parse Cloudflare response: {}", e)
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "error": format!("Request to Cloudflare failed: {}", e)
            })),
        )
            .into_response(),
    }
}
#[tokio::main]
async fn main() {
    // 加载 .env 文件（忽略文件不存在的错误）
    dotenvy::dotenv().ok();

    let cf_account_id = std::env::var("CF_ACCOUNT_ID").expect("CF_ACCOUNT_ID must be set");
    let cf_api_token = std::env::var("CF_API_TOKEN").expect("CF_API_TOKEN must be set");
    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    let state = Arc::new(AppState {
        cf_account_id,
        cf_api_token,
        client: reqwest::Client::new(),
    });

    let app = Router::new()
        // Cohere SDK posts to {base_url}/v1/rerank; accept both paths so we
        // match Hindsight's configured base_url (http://localhost:7070) as
        // well as any future clients that hit the bare /rerank alias.
        .route("/v1/rerank", post(rerank))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("Listening on {}", addr);
    axum::serve(listener, app).await.unwrap();
}
