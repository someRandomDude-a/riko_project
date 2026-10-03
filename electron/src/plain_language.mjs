// Reader-first guidance. Keep full implementation notes behind Technical details.
const help={
 'runtime.n_ctx':'How much text the model can use in one reply. Larger values use more memory.',
 'runtime.max_output_tokens':'The longest reply the model can produce. This also includes its internal reasoning.',
 'runtime.temperature':'Lower values give more consistent replies. Higher values add variety.',
 'runtime.pause_background_on_live':'Pause background thinking while you chat for faster replies.',
 'runtime.parallel_slots':'How many model requests can run at once. One is reserved for your chat.',
 'runtime.hf_repo_id':'The Hugging Face repository that contains your model.',
 'runtime.hf_filename':'Choose a model file from the selected repository.',
 'runtime.hf_revision':'The version of the repository to use.',
 'runtime.model_path':'Choose a model file on this computer.',
 'runtime.provider':'Choose the service that runs your chat model.',
 'runtime.n_gpu_layers':'Run more model layers on the GPU for speed, or fewer to save GPU memory. Use −1 for all layers.',
 'runtime.flash_attn':'Use a faster, lower-memory attention method when your model and GPU support it.',
 'runtime.type_k':'Choose the precision used to store remembered model context. Lower precision uses less memory.',
 'runtime.type_v':'Choose the precision used to store context values. Lower precision uses less memory.',
 'runtime.kv_unified':'Share one context cache across chat and background requests.',
 'runtime.n_batch':'How much prompt text to process at a time. Larger batches can use more memory.',
 'runtime.n_ubatch':'The batch size the model processes in a single step. Keep it at or below the prompt batch size.',
 'runtime.warmup':'Load models and test the voice service at startup. This does not start your microphone.',
 'memory.context_window_tokens':'The space available for your conversation and saved memories. Leave room for the reply.',
 'memory.token_budget':'How much space recalled memories can use in each prompt.',
 'tools.best_fit_inputs':'Fix clear spelling mistakes in supported tool choices. This never gives tools extra access.',
 'tools.require_approval':'Ask before running tools that do not have their own permission rule.',
 'voice.live_transcript_interval_seconds':'How often to update the text while you speak. Faster updates use more processing.',
 'voice.input_device':'Choose which microphone to use.',
 'voice.mode':'Choose how to start listening: a wake name, a button, or continuously.',
 'voice.wake_word':'The short name you say to start a conversation.',
 'voice.wake_threshold':'Higher values reduce false wake-ups but may miss your wake name.',
 'voice.follow_up_seconds':'How long to keep listening after a reply.',
 'avatar.enabled':'Show the desktop character.',
 'avatar.model':'Choose your desktop character model.',
 'avatar.camera.fov':'Change the camera’s viewing angle. The whole character stays in view.',
 'speech.max_words':'Aim to send this many words to the voice service at a time. Sentences are not cut in half.',
 'speech.split_window_words':'Look for a sentence break near the speech length limit.',
 'speech.split_priority':'Choose which punctuation marks to prefer when splitting spoken replies.',
 'wake_feedback.rules':'Choose the sounds and animations used when the character wakes up.',
 'animation.walk_speed':'How fast the character moves across the screen, in pixels per second.',
};
const labels={'runtime.provider':'Model service','runtime.n_ctx':'Conversation length (tokens)','runtime.max_output_tokens':'Maximum reply length (tokens)','runtime.temperature':'Reply variety','runtime.kv_unified':'Share conversation cache','runtime.n_gpu_layers':'Model layers on GPU','runtime.parallel_slots':'Concurrent requests','runtime.pause_background_on_live':'Pause background work while chatting','speech.max_words':'Spoken segment length (words)','speech.split_window_words':'Sentence-break search (words)','sovits_ping_config.url':'Voice service address','memory.context_window_tokens':'Conversation prompt budget (tokens)'};
export function simpleLabel(field){return labels[field.path]||field.label;}
export function simpleHelp(field){
 if(help[field.path])return help[field.path];
 if(!field.help)return field.restart===false?'Applies when you save.':'Save this change, then restart the backend.';
 // Use the first complete sentence only when it is short and free of internals.
 const first=field.help.match(/^.*?[.!?](?:\s|$)/)?.[0]?.trim();
 if(first&&first.length<130&&!/\b(KV|JSON|YAML|GGUF|CPU|CUDA|provenance|encoder|llama|tokenizer|override)\b/i.test(first))return first;
 if(field.options)return 'Choose an option from the list. Open Technical details for guidance.';
 if(field.file)return 'Choose a file or folder on this computer.';
 if(field.kind==='boolean')return 'Turn this feature on or off. Open Technical details for guidance.';
 return 'Set the value you want to use. Open Technical details for guidance.';
}
