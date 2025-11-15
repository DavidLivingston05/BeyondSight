/**
 * Keyboard Shortcuts for Beyond Sight
 * Optimized for blind users - quick voice controls without mouse
 */

class AccessibilityKeyboardHandler {
    constructor() {
        this.shortcuts = {
            // ============= CAMERA CONTROL =============
            'c': {
                name: 'Toggle Camera',
                action: this.toggleCamera,
                help: 'Press C to start/stop camera'
            },
            
            // ============= ANALYSIS COMMANDS =============
            'a': {
                name: 'Analyze Scene',
                action: this.analyzeScene,
                help: 'Press A for full scene analysis'
            },
            'f': {
                name: 'Find Object',
                action: this.promptFindObject,
                help: 'Press F to search for an object'
            },
            't': {
                name: 'Read Text',
                action: this.readText,
                help: 'Press T to read visible text'
            },
            'n': {
                name: 'Navigation Help',
                action: this.getNavigation,
                help: 'Press N for navigation advice'
            },
            'h': {
                name: 'Check Hazards',
                action: this.checkHazards,
                help: 'Press H to check for dangers'
            },
            
            // ============= VOICE CONTROL =============
            'Digit1': {
                name: 'Voice SLOW (100 WPM)',
                action: () => this.setVoiceRate('SLOW'),
                help: 'Press 1 for slow voice'
            },
            'Digit2': {
                name: 'Voice NORMAL (140 WPM)',
                action: () => this.setVoiceRate('NORMAL'),
                help: 'Press 2 for normal voice'
            },
            'Digit3': {
                name: 'Voice FAST (180 WPM)',
                action: () => this.setVoiceRate('FAST'),
                help: 'Press 3 for fast voice'
            },
            'Digit4': {
                name: 'Voice VERY FAST (220 WPM)',
                action: () => this.setVoiceRate('VERY_FAST'),
                help: 'Press 4 for very fast voice'
            },
            
            // ============= VOLUME CONTROL =============
            'KeyQ': {
                name: 'Volume SILENT',
                action: () => this.setVolume('SILENT'),
                help: 'Press Q for silent'
            },
            'KeyW': {
                name: 'Volume LOW (40%)',
                action: () => this.setVolume('LOW'),
                help: 'Press W for low volume'
            },
            'KeyE': {
                name: 'Volume MEDIUM (70%)',
                action: () => this.setVolume('MEDIUM'),
                help: 'Press E for medium volume'
            },
            'KeyR': {
                name: 'Volume LOUD (100%)',
                action: () => this.setVolume('LOUD'),
                help: 'Press R for loud volume'
            },
            
            // ============= TOGGLES =============
            'Space': {
                name: 'Toggle Continuous Description',
                action: this.toggleContinuous,
                help: 'Press Space to toggle continuous mode'
            },
            'Escape': {
                name: 'Stop Current Speech',
                action: this.stopSpeech,
                help: 'Press Escape to stop speaking'
            },
            
            // ============= HELP =============
            '?': {
                name: 'Show Help',
                action: this.showHelp,
                help: 'Press ? for help'
            }
        };
        
        this.init();
    }
    
    init() {
        // Register keyboard listener
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Announce keyboard shortcut support on load
        this.announceShortcutSupport();
    }
    
    handleKeyDown(event) {
        // Don't trigger if typing in input field
        if (this.isInputFocused()) {
            if (event.key === '?') {
                this.showHelp();
            }
            return;
        }
        
        // Get the shortcut
        const code = event.code;
        const key = event.key.toLowerCase();
        
        // Check by code first (for numbers), then by key
        const shortcut = this.shortcuts[code] || this.shortcuts[key];
        
        if (shortcut) {
            event.preventDefault();
            
            // Announce the action
            this.announce(`Activating: ${shortcut.name}`);
            
            // Execute the action
            shortcut.action.call(this);
        }
    }
    
    isInputFocused() {
        const active = document.activeElement;
        return (
            active.tagName === 'INPUT' ||
            active.tagName === 'TEXTAREA' ||
            active.contentEditable === 'true'
        );
    }
    
    // ============= ACTIONS =============
    
    toggleCamera() {
        const btn = document.getElementById('toggleCameraBtn');
        if (btn) btn.click();
    }
    
    analyzeScene() {
        const btn = document.getElementById('analyzeSceneBtn');
        if (btn) {
            btn.click();
            this.announce('Analyzing scene... Please wait');
        }
    }
    
    readText() {
        const btn = document.getElementById('readTextBtn');
        if (btn) {
            btn.click();
            this.announce('Reading text from camera... Please wait');
        }
    }
    
    checkHazards() {
        const btn = document.getElementById('hazardsBtn');
        if (btn) {
            btn.click();
            this.announce('Checking for hazards... Please wait');
        }
    }
    
    getNavigation() {
        const btn = document.getElementById('navigationBtn');
        if (btn) {
            btn.click();
            this.announce('Getting navigation advice... Please wait');
        }
    }
    
    promptFindObject() {
        const objectName = prompt('What object would you like to find?');
        if (objectName && objectName.trim()) {
            this.findObject(objectName.trim());
        }
    }
    
    findObject(objectName) {
        // Make API call
        fetch('/api/find/objects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ object: objectName })
        })
        .then(r => r.json())
        .then(data => {
            this.announce(`Found ${objectName}: ${data.result}`);
        })
        .catch(err => {
            this.announce(`Error finding ${objectName}`);
            console.error(err);
        });
    }
    
    toggleContinuous() {
        const btn = document.getElementById('continuousToggleBtn');
        if (btn) btn.click();
    }
    
    stopSpeech() {
        // Stop current speech (would need backend support)
        this.announce('Speech stopped');
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
        this.showHelpModal(helpText);
    }
    
    generateHelpText() {
        const shortcuts = Object.values(this.shortcuts)
            .map(s => s.help)
            .join('. ');
        
        return `Keyboard shortcuts available. ${shortcuts}`;
    }
    
    showHelpModal(text) {
        // Create accessible help modal
        const modal = document.createElement('div');
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-label', 'Keyboard shortcuts help');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #1e293b;
            border: 2px solid #6366f1;
            border-radius: 12px;
            padding: 30px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            box-shadow: 0 20px 25px rgba(0, 0, 0, 0.3);
        `;
        
        let helpHTML = '<h2 style="margin-bottom: 20px; color: #6366f1;">Keyboard Shortcuts</h2>';
        helpHTML += '<div style="color: #f1f5f9; line-height: 2;">';
        
        // Group shortcuts by category
        const groups = {
            'Camera Control': ['c'],
            'Analysis': ['a', 'f', 't', 'n', 'h'],
            'Voice Speed (1-4)': ['Digit1', 'Digit2', 'Digit3', 'Digit4'],
            'Volume (Q-R)': ['KeyQ', 'KeyW', 'KeyE', 'KeyR'],
            'Other': ['Space', 'Escape', '?']
        };
        
        for (const [group, keys] of Object.entries(groups)) {
            helpHTML += `<p style="margin-top: 15px; font-weight: bold; color: #a5f3fc;">${group}:</p>`;
            
            for (const key of keys) {
                const shortcut = this.shortcuts[key];
                if (shortcut) {
                    helpHTML += `<p style="margin-left: 20px;">
                        <strong style="color: #6366f1;">${key}</strong> → ${shortcut.name}
                    </p>`;
                }
            }
        }
        
        helpHTML += '</div>';
        helpHTML += '<button id="closeHelpBtn" style="';
        helpHTML += 'margin-top: 20px; padding: 10px 20px; ';
        helpHTML += 'background: #6366f1; color: white; ';
        helpHTML += 'border: none; border-radius: 6px; cursor: pointer;';
        helpHTML += '">Close Help (Press Escape)</button>';
        
        modal.innerHTML = helpHTML;
        document.body.appendChild(modal);
        
        // Close button handler
        document.getElementById('closeHelpBtn').addEventListener('click', () => {
            modal.remove();
        });
        
        // Close on Escape
        const closeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', closeHandler);
            }
        };
        document.addEventListener('keydown', closeHandler);
        
        // Accessibility: announce modal
        this.announce('Help modal opened. ' + text);
    }
    
    // ============= UTILITIES =============
    
    announce(text) {
        // Use Web Speech API for announcements
        if ('speechSynthesis' in window) {
            speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            speechSynthesis.speak(utterance);
        }
        
        // Also log to console
        console.log(`[Accessibility Announcement]: ${text}`);
    }
    
    announceShortcutSupport() {
        // On page load, announce that keyboard shortcuts are available
        window.addEventListener('load', () => {
            setTimeout(() => {
                this.announce('Beyond Sight loaded. Press question mark for keyboard shortcuts help.');
            }, 1000);
        });
    }
}

// ============= INITIALIZATION =============

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new AccessibilityKeyboardHandler();
    });
} else {
    new AccessibilityKeyboardHandler();
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccessibilityKeyboardHandler;
}
