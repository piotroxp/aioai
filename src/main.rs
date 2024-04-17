use ndarray::{Array, Array2, Array1, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use std::fs;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn load_weights(path: &str) -> Result<NeuralNetwork, serde_json::Error> {
    let data = fs::read_to_string(path).expect("Unable to read file");
    let nn = serde_json::from_str(&data)?;
    Ok(nn)
}

fn save_weights(nn: &NeuralNetwork, path: &str) -> Result<(), serde_json::Error> {
    let serialized = serde_json::to_string_pretty(nn)?;
    let mut file = File::create(path).expect("Unable to create file");
    file.write_all(serialized.as_bytes()).expect("Unable to write data");
    Ok(())
}

fn mean_squared_error(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let diff = predictions - targets;
    diff.mapv(|x| x.powi(2)).mean().unwrap()
}

#[derive(Serialize, Deserialize)]
struct NeuralNetwork {
    pub input_weights: Array2<f64>,
    pub output_weights: Array2<f64>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let input_weights = Array::random_using((input_size, hidden_size), Uniform::new(-1., 1.), &mut rng);
        let output_weights = Array::random_using((hidden_size, output_size), Uniform::new(-1., 1.), &mut rng);

        NeuralNetwork {
            input_weights,
            output_weights,
            learning_rate,
        }
    }

    fn forward(&self, inputs: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden_input = inputs.dot(&self.input_weights);
        let hidden_output = hidden_input.mapv(sigmoid);
        let final_input = hidden_output.dot(&self.output_weights);
        let final_output = final_input.mapv(sigmoid);
        (hidden_output, final_output)
    }

    fn train(&mut self, inputs: &Array1<f64>, targets: &Array1<f64>) {
        let (hidden_outputs, outputs) = self.forward(inputs);
    
        let output_errors = targets - &outputs;
        let output_delta = output_errors * outputs.mapv(sigmoid_derivative);
    
        let hidden_errors = output_delta.dot(&self.output_weights.t());
        let hidden_delta = hidden_errors * hidden_outputs.mapv(sigmoid_derivative);
    
        self.output_weights = &self.output_weights + &(&hidden_outputs.insert_axis(Axis(1)).dot(&output_delta.insert_axis(Axis(0))) * self.learning_rate);
        self.input_weights = &self.input_weights + &(&inputs.to_owned().insert_axis(Axis(1)).dot(&hidden_delta.insert_axis(Axis(0))) * self.learning_rate);
    }
    
}

fn main() {
    let mut nn = NeuralNetwork::new(4, 2, 5, 0.1);
    let inputs = Array::from_vec(vec![0.05, 10.0,2.4,29.0]);
    let targets = Array::from_vec(vec![1.0,2.0,0.1,0.01,10.0]);

    for _ in 0..10000 {
        nn.train(&inputs, &targets);
    }

    println!("Inputs: {:?}", inputs);
    let (_, predictions) = nn.forward(&inputs);
    println!("Weights: {:?}", nn.output_weights);
    println!("Predictions: {:?}", predictions);
    
    if let Err(e) = save_weights(&nn, "weights.json") {
        println!("Error saving weights: {}", e);
    }
}
