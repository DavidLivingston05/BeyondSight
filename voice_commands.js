/**
 * Voice Commands for Beyond Sight
 * Complete hands-free operation using Web Speech API
 */

class VoiceCommandHandler {
    constructor() {
        // Web Speech API setup
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error('Speech Recognition not supported');
            this.supported = false;
            return;
        }
        
        this.recognition = new SpeechRecognition();
        this.supported = true;
        this.isListening = false;
        this.isContinuousMode = false;
        
        // Configure recognition
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        
        // Voice command definitions
        this.commands = {
            // ============= CAMERA CONTROL =============
            'start camera': {
                keywords: ['start camera', 'turn on camera', 'activate camera'],
                action: () => this.executeCommand('startCamera'),
                help: 'Start the camera'
            },
            'stop camera': {
                keywords: ['stop camera', 'turn off camera', 'deactivate camera'],
                action: () => this.executeCommand('stopCamera'),
                help: 'Stop the camera'
            },
            
            // ============= ANALYSIS COMMANDS =============
            'analyze scene': {
                keywords: ['analyze scene', 'full analysis', 'describe everything', 'what do i see'],
                action: () => this.executeCommand('analyzeScene'),
                help: 'Full scene analysis'
            },
            'find people': {
                keywords: ['find people', 'are there people', 'locate people', 'anyone here'],
                action: () => this.executeCommand('findPeople'),
                help: 'Find people in scene'
            },
            'find doors': {
                keywords: ['find doors', 'locate doors', 'where is the door', 'door location'],
                action: () => this.executeCommand('findDoors'),
                help: 'Find doors'
            },
            'find chairs': {
                keywords: ['find chairs', 'locate chairs', 'where are the chairs', 'are there seats'],
                action: () => this.executeCommand('findChairs'),
                help: 'Find chairs/seats'
            },
            'read text': {
                keywords: ['read text', 'read what you see', 'what does it say', 'text recognition'],
                action: () => this.executeCommand('readText'),
                help: 'Read visible text'
            },
            'navigation': {
                keywords: ['navigation', 'guide me', 'how do i navigate', 'navigate', 'help me navigate'],
                action: () => this.executeCommand('getNavigation'),
                help: 'Get navigation guidance'
            },
            'check hazards': {
                keywords: ['check hazards', 'is it safe', 'any dangers', 'detect hazards', 'warning'],
                action: () => this.executeCommand('checkHazards'),
                help: 'Check for hazards'
            },
            
            // ============= VOICE SPEED =============
            'slow voice': {
                keywords: ['slow voice', 'speak slower', 'slow down', 'voice slow'],
                action: () => this.setVoiceRate('SLOW'),
                help: 'Set voice to slow'
            },
            'normal voice': {
                keywords: ['normal voice', 'normal speed', 'voice normal'],
                action: () => this.setVoiceRate('NORMAL'),
                help: 'Set voice to normal'
            },
            'fast voice': {
                keywords: ['fast voice', 'speak faster', 'speed up', 'voice fast'],
                action: () => this.setVoiceRate('FAST'),
                help: 'Set voice to fast'
            },
            'very fast voice': {
                keywords: ['very fast voice', 'very fast', 'maximum speed', 'voice very fast'],
                action: () => this.setVoiceRate('VERY_FAST'),
                help: 'Set voice to very fast'
            },
            
            // ============= VOLUME CONTROL =============
            'silent': {
                keywords: ['mute', 'silent', 'be quiet', 'no sound'],
                action: () => this.setVolume('SILENT'),
                help: 'Mute volume'
            },
            'low volume': {
                keywords: ['low volume', 'quiet', 'low', 'volume low'],
                action: () => this.setVolume('LOW'),
                help: 'Set volume to low'
            },
            'medium volume': {
                keywords: ['medium volume', 'medium', 'middle', 'volume medium'],
                action: () => this.setVolume('MEDIUM'),
                help: 'Set volume to medium'
            },
            'loud': {
                keywords: ['loud', 'loud volume', 'maximum', 'turn up', 'louder', 'volume loud'],
                action: () => this.setVolume('LOUD'),
                help: 'Set volume to loud'
            },
            
            // ============= TOGGLES =============
            'toggle continuous': {
                keywords: ['continuous', 'toggle continuous', 'continuous mode', 'keep describing'],
                action: () => this.executeCommand('toggleContinuous'),
                help: 'Toggle continuous description'
            },
            'reset chat': {
                keywords: ['reset chat', 'clear conversation', 'new conversation'],
                action: () => this.executeCommand('resetChat'),
                help: 'Reset conversation'
            },
            
            // ============= HELP =============
            'help': {
                keywords: ['help', 'commands', 'what can i do', 'voice commands'],
                action: () => this.showHelp(),
                help: 'Show voice command help'
            },
            'stop listening': {
                keywords: ['stop listening', 'stop listening', 'stop voice'],
                action: () => this.stopListening(),
                help: 'Stop voice recognition'
            }
        };
        
        this.init();
    }
    
    init() {
        // Setup recognition event handlers
        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateListeningUI(true);
            console.log('✅ Voice recognition STARTED - microphone active');
            this.updateTranscript('Listening for your voice...', true);
        };
        
        this.recognition.onresult = (event) => {
            console.log(`📊 onresult fired - results count: ${event.results.length}, resultIndex: ${event.resultIndex}`);
            
            let interimTranscript = '';
            let finalTranscript = '';
            let isFinal = false;
            
            // Collect all results since last event
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript.trim();
                const confidence = event.results[i][0].confidence;
                const isFinalResult = event.results[i].isFinal;
                
                console.log(`  Result[${i}]: "${transcript}" | Confidence: ${(confidence * 100).toFixed(1)}% | Final: ${isFinalResult}`);
                
                if (isFinalResult) {
                    // Only process transcripts with reasonable confidence
                    if (confidence > 0.5 || transcript.length > 2) {
                        finalTranscript += transcript + ' ';
                        isFinal = true;
                        console.log(`✓ Final ACCEPTED: "${transcript}"`);
                    } else {
                        console.log(`✗ Final REJECTED (low confidence): "${transcript}"`);
                    }
                } else {
                    interimTranscript += transcript;
                    console.log(`~ Interim: "${transcript}"`);
                }
            }
            
            // Show interim results
            if (interimTranscript && !isFinal) {
                this.updateTranscript(interimTranscript, true);
            }
            
            // Process final transcript
            if (finalTranscript) {
                console.log(`🎯 Processing final command: "${finalTranscript.trim()}"`);
                this.processVoiceCommand(finalTranscript.trim());
                this.updateTranscript(finalTranscript, false);
                
                // Restart in continuous mode
                if (this.isContinuousMode && this.isListening) {
                    setTimeout(() => {
                        try {
                            console.log('🔄 Restarting recognition in continuous mode...');
                            this.recognition.start();
                        } catch (e) {
                            console.warn('⚠️ Could not restart recognition:', e);
                        }
                    }, 300);
                }
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('❌ Recognition ERROR:', event.error);
            this.announce(`Voice recognition error: ${event.error}`);
            
            // Auto-restart on certain errors
            if (event.error === 'no-speech' || event.error === 'audio-capture') {
                console.log('⚠️ Attempting to restart after error...');
                setTimeout(() => {
                    try {
                        this.recognition.start();
                    } catch (e) {
                        console.error('Failed to restart:', e);
                    }
                }, 1000);
            }
        };
        
        this.recognition.onend = () => {
            console.log('⏹️ Voice recognition ENDED - isContinuousMode:', this.isContinuousMode);
            this.isListening = false;
            this.updateListeningUI(false);
            
            // Restart listening if in continuous mode
            if (this.isContinuousMode) {
                try {
                    setTimeout(() => {
                        console.log('🔄 Restarting voice recognition after end...');
                        this.recognition.start();
                    }, 300);
                } catch (error) {
                    console.error('Error restarting recognition:', error);
                }
            }
        };
        
        // Create floating voice button
        this.createFloatingVoiceButton();
    }
    
    createFloatingVoiceButton() {
        if (!this.supported) return;
        
        const btn = document.createElement('button');
        btn.id = 'voiceCommandBtn';
        btn.setAttribute('aria-label', 'Start voice commands');
        btn.innerHTML = '🎤';
        btn.style.cssText = `
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: 2px solid rgba(99, 102, 241, 0.3);
            color: white;
            font-size: 28px;
            cursor: pointer;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        `;
        
        btn.onmouseover = () => {
            btn.style.transform = 'scale(1.1)';
            btn.style.boxShadow = '0 15px 35px rgba(99, 102, 241, 0.6)';
        };
        btn.onmouseout = () => {
            btn.style.transform = 'scale(1)';
            btn.style.boxShadow = '0 10px 25px rgba(99, 102, 241, 0.4)';
        };
        
        btn.onclick = (e) => {
            e.preventDefault();
            this.toggleListening();
        };
        document.body.appendChild(btn);
        
        // Create listening indicator
        this.createListeningIndicator();
    }
    
    createListeningIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'voiceListeningIndicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50px;
            background: rgba(99, 102, 241, 0.9);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            display: none;
            z-index: 998;
            max-width: 300px;
            word-wrap: break-word;
        `;
        
        document.body.appendChild(indicator);
    }
    
    updateListeningUI(isListening) {
        const btn = document.getElementById('voiceCommandBtn');
        if (!btn) return;
        
        if (isListening) {
            btn.style.background = 'linear-gradient(135deg, #ef4444, #f97316)';
            btn.style.animation = 'pulse 1.5s infinite';
            btn.innerHTML = '🎤 Listening...';
            btn.style.fontSize = '14px';
            btn.style.fontWeight = 'bold';
            this.announce('Listening for voice commands. Say "help" for available commands.');
        } else {
            btn.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
            btn.style.animation = 'none';
            btn.innerHTML = '🎤';
            btn.style.fontSize = '28px';
            btn.style.fontWeight = 'normal';
        }
    }
    
    updateTranscript(text, isInterim = false) {
        const indicator = document.getElementById('voiceListeningIndicator');
        if (!indicator) return;
        
        if (isInterim) {
            indicator.style.opacity = '0.6';
            indicator.textContent = `Listening... "${text}"`;
        } else {
            indicator.style.opacity = '1';
            indicator.textContent = `You said: "${text}"`;
        }
        
        console.log(`[${isInterim ? 'INTERIM' : 'FINAL'}]: "${text}"`);
    }
    
    toggleListening() {
        if (!this.supported) {
            alert('Speech Recognition is not supported in your browser');
            return;
        }
        
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }
    
    startListening() {
        try {
            // Make sure we're not already listening
            if (this.isListening) {
                console.warn('⚠️ Already listening, skipping start');
                return;
            }

            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.isContinuousMode = true;
            this.recognition.start();
            console.log('🟢 Voice recognition started - listening continuously...');
            console.log('🎤 Speak your command now...');
        } catch (error) {
            console.error('❌ Error starting voice recognition:', error.message);
            this.announce(`Error starting microphone: ${error.message}`);
        }
    }
    
    stopListening() {
        this.isContinuousMode = false;
        try {
            this.recognition.continuous = false;
            this.recognition.stop();
            console.log('Voice recognition stopped');
        } catch (error) {
            console.error('Error stopping voice recognition:', error);
        }
    }
    
    processVoiceCommand(transcript) {
         const text = transcript.toLowerCase().trim();
         console.log(`Processing voice command: "${text}"`);
         
         // Skip very short or empty transcripts (noise filtering)
         if (text.length < 3) {
             console.warn(`Skipping short transcript: "${text}"`);
             return;
         }
         
         // Find best matching command with confidence scoring
         let bestMatch = null;
         let bestScore = 0;
         const MIN_CONFIDENCE = 0.3;
         
         for (const [commandName, commandData] of Object.entries(this.commands)) {
             for (const keyword of commandData.keywords) {
                  // Calculate match score (keywords are more important than partial matches)
                  let score = 0;
                  
                  // Exact keyword match gets full score
                  if (text === keyword) {
                      score = 1.0;
                  }
                  // Keyword as substring gets partial score
                  else if (text.includes(keyword)) {
                      score = 0.9;
                  }
                  // Check if keyword is close to the transcript (fuzzy matching)
                  else {
                      const words = text.split(' ');
                      const keywordWords = keyword.split(' ');
                      const matchCount = keywordWords.filter(kw => 
                          words.some(w => w.includes(kw) || kw.includes(w))
                      ).length;
                      score = matchCount / keywordWords.length;
                  }
                  
                  if (score > bestScore) {
                      bestScore = score;
                      bestMatch = { commandName, commandData, keyword };
                  }
              }
         }
         
         // Execute best match if confidence is high enough
         if (bestMatch && bestScore >= MIN_CONFIDENCE) {
             console.log(`Matched command: ${bestMatch.commandName} (score: ${bestScore.toFixed(2)})`);
             this.announce(`Executing: ${bestMatch.commandName}`);
             bestMatch.commandData.action();
             return;
         }
         
         // If no good match found, provide feedback
         console.warn(`No matching command found for: "${text}"`);
         this.announce(`Didn't catch that. Say help for available voice commands.`);
     }
    
    executeCommand(functionName) {
        // Call the main application function
        if (typeof window[functionName] === 'function') {
            window[functionName]();
        } else {
            console.error(`Function ${functionName} not found`);
            this.announce(`Error executing ${functionName}`);
        }
    }
    
    setVoiceRate(rate) {
        fetch('/api/voice/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rate: rate })
        })
        .then(r => r.json())
        .then(data => {
            const speedMap = {
                'SLOW': 'slow',
                'NORMAL': 'normal',
                'FAST': 'fast',
                'VERY_FAST': 'very fast'
            };
            this.announce(`Voice speed set to ${speedMap[rate]}`);
        })
        .catch(err => {
            this.announce('Error setting voice speed');
            console.error(err);
        });
    }
    
    setVolume(volume) {
        fetch('/api/voice/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ volume: volume })
        })
        .then(r => r.json())
        .then(data => {
            const volumeMap = {
                'SILENT': 'silent',
                'LOW': 'low',
                'MEDIUM': 'medium',
                'LOUD': 'loud'
            };
            this.announce(`Volume set to ${volumeMap[volume]}`);
        })
        .catch(err => {
            this.announce('Error setting volume');
            console.error(err);
        });
    }
    
    showHelp() {
        const helpText = this.generateHelpText();
        this.announce(helpText);
        this.showHelpModal();
    }
    
    generateHelpText() {
        const categories = {
            'Camera': ['Start camera', 'Stop camera'],
            'Analysis': ['Analyze scene', 'Find people', 'Find doors', 'Find chairs', 'Read text', 'Navigation'],
            'Voice Speed': ['Slow voice', 'Normal voice', 'Fast voice', 'Very fast voice'],
            'Volume': ['Silent', 'Low volume', 'Medium volume', 'Loud'],
            'Control': ['Toggle continuous', 'Reset chat'],
            'Help': ['Help', 'Stop listening']
        };
        
        let helpText = 'Voice commands available. ';
        for (const [category, commands] of Object.entries(categories)) {
            helpText += `${category}: ${commands.join(', ')}. `;
        }
        return helpText;
    }
    
    showHelpModal() {
        const modal = document.createElement('div');
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-label', 'Voice commands help');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #1e293b;
            border: 3px solid #6366f1;
            border-radius: 16px;
            padding: 40px;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
            animation: slideDown 0.3s ease;
        `;
        
        let helpHTML = '<h2 style="margin-bottom: 30px; color: #6366f1; font-size: 28px;">🎤 Voice Commands Help</h2>';
        helpHTML += '<div style="color: #f1f5f9; line-height: 2.2; font-size: 16px;">';
        
        const categories = {
            '📷 Camera Control': [
                { cmd: 'Start camera', ex: '"Start camera"' },
                { cmd: 'Stop camera', ex: '"Stop camera"' }
            ],
            '🔍 Analysis': [
                { cmd: 'Analyze scene', ex: '"What do I see"' },
                { cmd: 'Find people', ex: '"Are there people here"' },
                { cmd: 'Find doors', ex: '"Where is the door"' },
                { cmd: 'Find chairs', ex: '"Are there seats"' },
                { cmd: 'Read text', ex: '"Read what you see"' },
                { cmd: 'Navigation', ex: '"Guide me"' },
                { cmd: 'Check hazards', ex: '"Is it safe"' }
            ],
            '🔊 Voice Speed': [
                { cmd: 'Slow voice', ex: '"Speak slower"' },
                { cmd: 'Normal voice', ex: '"Normal speed"' },
                { cmd: 'Fast voice', ex: '"Speed up"' },
                { cmd: 'Very fast voice', ex: '"Maximum speed"' }
            ],
            '🔉 Volume': [
                { cmd: 'Silent', ex: '"Mute"' },
                { cmd: 'Low volume', ex: '"Quiet"' },
                { cmd: 'Medium volume', ex: '"Medium"' },
                { cmd: 'Loud', ex: '"Turn up volume"' }
            ]
        };
        
        for (const [category, commands] of Object.entries(categories)) {
            helpHTML += `<p style="margin-top: 20px; font-weight: bold; color: #a5f3fc; font-size: 18px;">${category}</p>`;
            
            for (const { cmd, ex } of commands) {
                helpHTML += `<p style="margin-left: 20px; margin-top: 10px;">
                    <strong style="color: #6366f1;">${cmd}</strong><br>
                    <span style="font-size: 14px; color: #cbd5e1;">Example: ${ex}</span>
                </p>`;
            }
        }
        
        helpHTML += '<p style="margin-top: 20px; color: #10b981; font-weight: bold;">💡 Tip: Say "Help" anytime to see available commands</p>';
        helpHTML += '</div>';
        helpHTML += '<button id="closeVoiceHelpBtn" style="';
        helpHTML += 'margin-top: 30px; padding: 12px 24px; ';
        helpHTML += 'background: #6366f1; color: white; ';
        helpHTML += 'border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold;';
        helpHTML += '">Close (Press Escape)</button>';
        
        modal.innerHTML = helpHTML;
        document.body.appendChild(modal);
        
        const closeBtn = document.getElementById('closeVoiceHelpBtn');
        closeBtn.addEventListener('click', () => modal.remove());
        
        const closeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', closeHandler);
            }
        };
        document.addEventListener('keydown', closeHandler);
    }
    
    announce(text) {
         // Use Web Speech API for immediate feedback
         if ('speechSynthesis' in window) {
             speechSynthesis.cancel();
             const utterance = new SpeechSynthesisUtterance(text);
             utterance.rate = 1.0;
             utterance.pitch = 1.0;
             utterance.volume = 1.0;
             utterance.lang = 'en-US';
             
             // Find English voice
             const voices = speechSynthesis.getVoices();
             const englishVoice = voices.find(voice => voice.lang.includes('en'));
             if (englishVoice) {
                 utterance.voice = englishVoice;
             }
             
             speechSynthesis.speak(utterance);
         } else {
             console.warn('Speech Synthesis not supported');
         }
         
         console.log(`[Voice Announcement]: ${text}`);
         
         // Send via WebSocket if available
         if (typeof socket !== 'undefined' && socket.connected) {
             socket.emit('voice_feedback', { text });
         }
     }
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { 
            opacity: 1;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.4);
        }
        50% { 
            opacity: 0.7;
            box-shadow: 0 15px 35px rgba(239, 68, 68, 0.6);
        }
    }
    
    @keyframes slideDown {
        from {
            transform: translate(-50%, -60%);
            opacity: 0;
        }
        to {
            transform: translate(-50%, -50%);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// ============= INITIALIZATION =============

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.voiceCommandHandler = new VoiceCommandHandler();
    });
} else {
    window.voiceCommandHandler = new VoiceCommandHandler();
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceCommandHandler;
}
