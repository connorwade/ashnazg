use dotenv;
use llm::Model;
use neon::prelude::*;
use std::convert::Infallible;
use std::env;
use std::io::Write;
use std::path::PathBuf;

fn hello(mut cx: FunctionContext) -> JsResult<JsString> {
    Ok(cx.string("hello node"))
}

fn define_model() -> Box<dyn Model> {
    let architecture = llm::ModelArchitecture::Llama.to_string();
    let path = &get_model_path();
    let tokenizer = llm::TokenizerSource::Embedded;
    let params = llm::ModelParameters {
        prefer_mmap: true,
        ..Default::default()
    };
    let model = llm::load_dynamic(
        architecture.parse().ok(),
        path,
        tokenizer,
        params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load model from {path:?}: {err}"));

    model
}

fn get_model_path() -> PathBuf {
    dotenv::dotenv().ok();
    let model_path = env::var("MODEL_PATH").expect("MODEL_PATH must be set");
    let path = PathBuf::from(&model_path);
    return path;
}

fn get_language_model(mut cx: FunctionContext) -> JsResult<JsString> {
    use llm::KnownModel;
    let binding = define_model();
    let llama: &dyn llm::Model = binding.as_ref();

    let mut res = String::new();
    let mut buf = String::new();

    let mut session = llama.start_session(Default::default());
    let res = session.infer::<std::convert::Infallible>(
        llama,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: "Rust is a cool programming language because".into(),
            // ..Default::default()
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut Default::default(),
        inference_callback(&mut buf, &mut res),
    );

    let str = match res {
        Ok(result) => format!("\n\nInference stats:\n{result}"),
        Err(err) => format!("{err}"),
    };

    Ok(cx.string(str))
}

fn inference_callback<'a>(
    buf: &'a mut String,
    out_str: &'a mut String,
) -> impl FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, Infallible> + 'a {
    use llm::InferenceFeedback::Continue;
    use llm::InferenceFeedback::Halt;

    move |resp| match resp {
        llm::InferenceResponse::InferredToken(t) => {
            let mut reverse_buf = buf.clone();
            reverse_buf.push_str(t.as_str());

            if buf.is_empty() {
                out_str.push_str(&t);
            } else {
                out_str.push_str(&reverse_buf)
            }

            Ok(Continue)
        }

        llm::InferenceResponse::EotToken => Ok(Halt),
        _ => Ok(Continue),
    }
}

// fn get_language_model() -> Llama {
//     use std::path::PathBuf;
//     dotenv().ok();
//     let module_path = env::var("MODEL_PATH").expect("MODEL_PATH must be set");

//     llm::load::<Llama>(
//         &PathBuf::from(&module_path),
//         llm::TokenizerSource::Embedded,
//         Default::default(),
//         llm::load_progress_callback_stdout,
//     )
//     .unwrap_or_else(|err| panic!("Failed to load model from {module_path:?}: {err}"))
// }

// async fn converse(prompt: String) {
//     use llm::KnownModel;

//     let model = llm::Model;
//     let mut session = model.start_session(Default::default());

//     let character_name = "### Assistant";
//     let user_name = "### Human";
//     let persona = "A chat between a human and an assistant";
//     let mut history = format!(
//         "{character_name}:Hello - How may I help you today?\n\
//         {user_name}:What is the capital of France\n\
//         {character_name}:Paris is the capital of France.\n
//         "
//     );

//     let llama = &llm::InferenceRequest {
//         prompt: format!("{persona}\n{history}\n{character_name}:")
//             .as_str()
//             .into(),
//         parameters: &llm::InferenceParameters::default(),
//         play_back_previous_tokens: false,
//         maximum_token_count: None,
//     };
// }

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("hello", hello)?;
    cx.export_function("languageModel", get_language_model)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::define_model;

    #[test]
    fn load_model() {
        define_model();
    }
}
