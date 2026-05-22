pub type DataMatrix = Vec<Vec<f64>>;
pub type Labels = Vec<usize>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelParams {
    pub k: usize,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct PnnModel {
    pub x_train: DataMatrix,
    pub y_train: Labels,
    pub n_classes: usize,
    pub k_max: usize,
}
