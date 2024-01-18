use std::env;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut image_path_in = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image_path" => {
                image_path_in = args[i + 1].clone();
                i += 2;
            }
            _ => {
                eprintln!("Error: Unsupported flag or positional parameter {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    let start_time = Instant::now();
    let output = Command::new("accelerate")
        .args(&["launch", "--config_file", "1gpu.yaml", "test_mvdiffusion_seq.py", "--config", "configs/mvdiffusion-joint-ortho-6views.yaml", &format!("validation_dataset.root_dir={}", std::path::Path::new(&image_path_in).parent().unwrap().display()), &format!("validation_dataset.filepaths=[{}]", std::path::Path::new(&image_path_in).file_name().unwrap().to_str().unwrap())])
        .output()
        .expect("Failed to execute command");
    let elapsed_time = start_time.elapsed();
    println!("View generation: {} seconds", elapsed_time.as_secs());

    let job_name = std::path::Path::new(&image_path_in).file_stem().unwrap().to_str().unwrap();

    let start_time = Instant::now();
    let output = Command::new("python")
        .current_dir("instant-nsr-pl")
        .args(&["launch.py", "--config", "configs/neuralangelo-ortho-wmask.yaml", "--gpu", "0", "--train", &format!("dataset.root_dir=../outputs/cropsize-192-cfg3.0 dataset.scene={}", job_name)])
        .output()
        .expect("Failed to execute command");
    let elapsed_time = start_time.elapsed();
    println!("Nerf fit: {} seconds", elapsed_time.as_secs());
}
