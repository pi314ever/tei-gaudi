use hf_hub::api::{
    tokio::{ApiError, ApiRepo},
    RepoInfo,
};
use std::path::PathBuf;
use tracing::instrument;

// Old classes used other config names than 'sentence_bert_config.json'
pub const ST_CONFIG_NAMES: [&str; 7] = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
];

/// Parses a [`RepoInfo`] object for model weight files to download. Returns a non-empty vector of
/// model files that contains a model type str, if there are any.
#[instrument(skip_all)]
fn _weight_files_to_download<'a>(
    api_info: &'a RepoInfo,
    weight_file_type_str: &'a str,
) -> Option<Vec<&'a str>> {
    let ignored_file_contains = ["arguments", "args", "training", "medusa_lm_head"];
    let files: Vec<&str> = api_info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| {
            f.contains(weight_file_type_str)
                && f.split("/").count() == 1
                && ignored_file_contains.iter().all(|s| !f.contains(s))
        })
        .collect();
    if files.is_empty() {
        return None;
    }
    Some(files)
}

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();

    tracing::info!("Starting download");

    let model_root = api
        .get("config.json")
        .await?
        .parent()
        .unwrap()
        .to_path_buf();
    api.get("tokenizer.json").await?;

    let api_info = api.info().await?;
    let model_files = _weight_files_to_download(&api_info, "safetensors")
        .or_else(|| {
            tracing::warn!("`model.safetensors` not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            _weight_files_to_download(&api_info, "pytorch_model")
        }).expect("No model files found as `safetensors` or `pytorch_model`");

    // Download the model files
    for file_name in model_files {
        api.get(file_name).await?;
    }
    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}

#[instrument(skip_all)]
pub async fn download_pool_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let pool_config_path = api.get("1_Pooling/config.json").await?;
    Ok(pool_config_path)
}

#[instrument(skip_all)]
pub async fn download_st_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    // Try default path
    let err = match api.get(ST_CONFIG_NAMES[0]).await {
        Ok(st_config_path) => return Ok(st_config_path),
        Err(err) => err,
    };

    for name in &ST_CONFIG_NAMES[1..] {
        if let Ok(st_config_path) = api.get(name).await {
            return Ok(st_config_path);
        }
    }

    Err(err)
}
