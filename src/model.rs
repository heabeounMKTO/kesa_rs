use anyhow::{Error, Result};

use serde::{Deserialize, Serialize};

/// just use data.yaml used to train the model lol,
/// yes i steal my own code smh
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetInfo {
    pub names: Vec<String>,
    pub nc: i64,
    pub train: String,
    pub val: String,
    pub test: String,
}
