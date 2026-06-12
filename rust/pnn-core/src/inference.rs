use crate::types::{PnnModel, PosteriorDraw, SamplerConfig};

pub fn sample_posterior(_model: &PnnModel, _config: &SamplerConfig) -> Vec<PosteriorDraw> {
    Vec::new()
}
