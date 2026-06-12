use crate::types::{PnnModel, PosteriorDraw};

pub fn predict(_model: &PnnModel, _x_new: &[f64], _draws: &[PosteriorDraw]) -> Vec<f64> {
    Vec::new()
}
