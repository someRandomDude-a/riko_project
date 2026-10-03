// Central skin presets and tokens. No remote UI assets or fonts are required.
export const skins = ['solid', 'gradient', 'glass'];
export const colorPresets = {
  charcoal:{background:'#101116',surface:'#191b23',accent:'#b6a1ff',text:'#eaeaf1',muted:'#9b9fac',gradientStart:'#252338',gradientEnd:'#101116'},
  slate:{background:'#101820',surface:'#18242e',accent:'#81bacc',text:'#e6edf1',muted:'#96a9b5',gradientStart:'#183645',gradientEnd:'#101820'},
  forest:{background:'#111a17',surface:'#1c2922',accent:'#91c7a8',text:'#e6ede9',muted:'#99b0a3',gradientStart:'#223c31',gradientEnd:'#111a17'},
};
export const appearanceDefaults={...colorPresets.charcoal,skin:'glass',gradientAngle:135,glassOpacity:.72,windowOpacity:.86,boardOpacity:.8,glassBlur:24,cornerRadius:18,reduceMotion:false,advancedExplanations:false};
export function appearanceStyle(preferences) {
  const value={...appearanceDefaults,...preferences};
  return {'--bg':value.background,'--panel':value.surface,'--accent':value.accent,'--text':value.text,
    '--muted':value.muted,'--gradient-start':value.gradientStart,'--gradient-end':value.gradientEnd,
    '--gradient-angle':value.gradientAngle+'deg','--glass-alpha':value.glassOpacity,
    '--window-alpha':value.windowOpacity,'--board-alpha':value.boardOpacity,'--glass-blur':value.glassBlur+'px','--corner-radius':value.cornerRadius+'px'};
}
