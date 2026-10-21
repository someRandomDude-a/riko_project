// character_config.yaml as Electron reads it at launch (main.cjs): desktop.debug, desktop.shortcuts, the companion's name
// and, when the packaged app starts GPT-SoVITS, sovits_ping_config. The backend reads the same file with PyYAML's safe_load
// (YAML 1.1), so this reads it the same way rather than as YAML 1.2: an unquoted yes, no, on or off is a boolean, 017 is
// octal and 1:30 is 90, while y, n, 09, 1e3 and 2024-1-5 stay text, and a repeated key keeps its last value. Anything else
// Electron takes from the YAML belongs here too. tests/test_electron_settings.py checks these readings against PyYAML.
const YAML=require('yaml');
// PyYAML's implicit resolvers (yaml/resolver.py) where the yaml package's YAML 1.1 schema accepts more.
const PYYAML={
 true:/^(?:yes|Yes|YES|true|True|TRUE|on|On|ON)$/,false:/^(?:no|No|NO|false|False|FALSE|off|Off|OFF)$/,
 int:/^[-+]?(?:0|[1-9][0-9_]*)$/,intTime:/^[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+$/,
 float:/^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?)$/,  // a dot, and a signed exponent
 timestamp:/^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$/,
};
const both=(ours,theirs)=>({test:value=>ours.test(value)&&theirs.test(value)});  // a timestamp's resolve re-reads its own match
function pyyamlTags(tags){
 return tags.filter(tag=>tag.format!=='EXP').map(tag=>{
  const kind=tag.tag.slice('tag:yaml.org,2002:'.length);
  if(kind==='bool')return {...tag,test:PYYAML[tag.identify(true)?'true':'false']};
  if(kind==='int'&&!tag.format)return {...tag,test:PYYAML.int};
  if(kind==='int'&&tag.format==='TIME')return {...tag,test:PYYAML.intTime};
  if(kind==='float'&&!tag.format&&!tag.test.test('.nan'))return {...tag,test:PYYAML.float};  // not .inf and .nan's tag
  if(kind==='timestamp')return {...tag,test:both(PYYAML.timestamp,tag.test)};
  return tag;  // binary, octal and hex numbers, sexagesimal floats, .inf and .nan, null, merge keys: as PyYAML has them
 });
}
const mapping=value=>value&&typeof value==='object'&&!Array.isArray(value)&&!(value instanceof Date)?value:{};
function parseConfig(text){return mapping(YAML.parse(String(text),{version:'1.1',customTags:pyyamlTags,uniqueKeys:false}));}
// The name as the backend has it as text (str() of what PyYAML read: an unquoted Yes is True); anything else is no name.
const text=value=>typeof value==='string'?value:typeof value==='boolean'?(value?'True':'False'):typeof value==='number'&&Number.isFinite(value)?String(value):'';
function launchSettings(source){
 const config=parseConfig(source),desktop=mapping(config.desktop),preset=mapping(mapping(config.presets).default);
 // The companion's name as load_config takes it: presets.default.name, else the older top-level character_name.
 const name=Object.hasOwn(preset,'name')?preset.name:config.character_name;
 return {debug:desktop.debug===true,shortcuts:desktop.shortcuts,name:text(name),sovits:config.sovits_ping_config};
}
module.exports={parseConfig,launchSettings};
