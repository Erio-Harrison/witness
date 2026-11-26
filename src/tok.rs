//! Tokenizer of the model (HF tokenizer.json) and the token ids the FIM prompt needs.
use tokenizers::Tokenizer;

pub struct Tok {
    inner: Tokenizer,
    pub fim_prefix: i32,
    pub fim_suffix: i32,
    pub fim_middle: i32,
    pub fim_repo: i32,
    pub fim_sep: i32,
    /// End-of-generation tokens.
    pub eog: Vec<i32>,
}

impl Tok {
    pub fn load(tokenizer_json: &std::path::Path) -> Result<Tok, String> {
        let mut inner = Tokenizer::from_file(tokenizer_json).map_err(|e| format!("{}: {e}", tokenizer_json.display()))?;
        // Text from a client is text: "<|im_end|>" typed in a file must not become the control token.
        inner.set_encode_special_tokens(true);
        let id = |name: &str| inner.token_to_id(name).map(|i| i as i32).ok_or(format!("the tokenizer has no {name}"));
        Ok(Tok {
            fim_prefix: id("<|fim_prefix|>")?,
            fim_suffix: id("<|fim_suffix|>")?,
            fim_middle: id("<|fim_middle|>")?,
            fim_repo: id("<|repo_name|>")?,
            fim_sep: id("<|file_sep|>")?,
            eog: vec![id("<|endoftext|>")?, id("<|im_end|>")?, id("<|fim_pad|>")?, id("<|repo_name|>")?, id("<|file_sep|>")?],
            inner,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        match self.inner.encode(text, false) {
            Ok(e) => e.get_ids().iter().map(|&i| i as i32).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&ids, false).unwrap_or_default()
    }

    /// Incremental detokenizer: feed token ids, get the text that became complete. A token that
    /// ends inside a UTF-8 character yields "" and the character comes out with a later token.
    pub fn stream(&self) -> impl FnMut(i32) -> String + '_ {
        let mut stream = self.inner.decode_stream(false);
        move |id| stream.step(id as u32).ok().flatten().unwrap_or_default()
    }
}
