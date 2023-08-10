use crate::convert_label::convert::{ConvertSettings, ConvertTarget};


#[derive(Debug, Clone)]
pub enum KesaTask {
    KesaConvert,
    KesaLabel,
    KesaAugment
}

