# modules/integrations/claude_dialog.py
"""
Module: Claude Conversation Dialog
Purpose: PyQt6 dialog for interactive Claude conversations about trading analysis
Features: Real-time chat, export functionality, markdown rendering
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
    QLabel, QSplitter, QWidget, QFileDialog, QMessageBox,
    QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
integrations_dir = current_dir
modules_dir = os.path.dirname(integrations_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

from modules.integrations.claude_integration import ClaudeIntegration

logger = logging.getLogger(__name__)


class ClaudeWorker(QThread):
    """Worker thread for Claude API calls"""
    
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, claude_integration: ClaudeIntegration):
        super().__init__()
        self.claude = claude_integration
        self.message = ""
        self.conversation_id = None
        
    def set_message(self, message: str, conversation_id: Optional[str] = None):
        """Set message to send"""
        self.message = message
        self.conversation_id = conversation_id
        
    def run(self):
        """Run the worker"""
        try:
            self.progress_update.emit("Sending message to Claude...")
            response, _ = self.claude.send_message(
                self.message, 
                self.conversation_id
            )
            self.response_received.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ClaudeConversationDialog(QDialog):
    """Interactive dialog for Claude conversations about trading analysis"""
    
    def __init__(self, parent=None, initial_results: Dict[str, Any] = None,
                 context: Dict[str, Any] = None):
        super().__init__(parent)
        self.claude = ClaudeIntegration()
        self.current_conversation_id = None
        self.initial_results = initial_results
        self.context = context or {}
        self.worker = None
        
        self.init_ui()
        self.apply_dark_theme()
        
        # Start initial analysis if results provided
        if self.initial_results:
            QTimer.singleShot(100, self.analyze_initial_results)
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Claude Trading Analysis")
        self.setGeometry(200, 200, 1000, 700)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Claude Trading Analysis Assistant")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Export button
        self.export_btn = QPushButton("Export Conversation")
        self.export_btn.clicked.connect(self.export_conversation)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)
        
        # Clear button
        self.clear_btn = QPushButton("New Conversation")
        self.clear_btn.clicked.connect(self.new_conversation)
        header_layout.addWidget(self.clear_btn)
        
        layout.addLayout(header_layout)
        
        # Conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.conversation_display, 3)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Input section
        input_layout = QVBoxLayout()
        
        input_label = QLabel("Your Question:")
        input_layout.addWidget(input_label)
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText(
            "Ask about the analysis, request optimizations, or explore specific aspects..."
        )
        input_layout.addWidget(self.input_field)
        
        # Send button
        self.send_btn = QPushButton("Send to Claude")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout, 1)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00cc00;")
        layout.addWidget(self.status_label)
        
    def apply_dark_theme(self):
        """Apply dark theme to dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0d7377;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
            QLabel {
                color: #ffffff;
            }
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)
        
    def add_message_to_display(self, role: str, content: str):
        """Add message to conversation display with formatting"""
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add timestamp
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        cursor.insertText(f"\n[{timestamp}] ", self._get_timestamp_format())
        
        # Add role
        if role == "user":
            cursor.insertText("You: ", self._get_user_format())
        else:
            cursor.insertText("Claude: ", self._get_claude_format())
        
        # Add content
        cursor.insertText(content + "\n", self._get_content_format())
        
        # Scroll to bottom
        self.conversation_display.ensureCursorVisible()
        
    def _get_timestamp_format(self) -> QTextCharFormat:
        """Get format for timestamps"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#888888"))
        return fmt
        
    def _get_user_format(self) -> QTextCharFormat:
        """Get format for user label"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#00cc00"))
        fmt.setFontWeight(QFont.Weight.Bold)
        return fmt
        
    def _get_claude_format(self) -> QTextCharFormat:
        """Get format for Claude label"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#00aaff"))
        fmt.setFontWeight(QFont.Weight.Bold)
        return fmt
        
    def _get_content_format(self) -> QTextCharFormat:
        """Get format for message content"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ffffff"))
        return fmt
        
    def analyze_initial_results(self):
        """Analyze initial backtest results"""
        if not self.initial_results:
            return
            
        self.status_label.setText("Analyzing backtest results...")
        self.status_label.setStyleSheet("color: #ffcc00;")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Disable inputs
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        
        # Create worker
        self.worker = ClaudeWorker(self.claude)
        self.worker.response_received.connect(self.handle_initial_analysis)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.progress_update.connect(self.update_status)
        
        # Create conversation and format prompt
        self.current_conversation_id = self.claude.create_conversation(self.context)
        prompt = self.claude._format_backtest_prompt(self.initial_results, self.context)
        
        # Add initial message to display
        self.add_message_to_display("user", "Please analyze these backtesting results...")
        
        # Start analysis
        self.worker.set_message(prompt, self.current_conversation_id)
        self.worker.start()
        
    def handle_initial_analysis(self, response: str):
        """Handle initial analysis response"""
        self.add_message_to_display("assistant", response)
        
        # Re-enable controls
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.hide()
        
        self.status_label.setText("Analysis complete - ask follow-up questions")
        self.status_label.setStyleSheet("color: #00cc00;")
        
    def send_message(self):
        """Send message to Claude"""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
            
        # Add to display
        self.add_message_to_display("user", message)
        
        # Clear input
        self.input_field.clear()
        
        # Disable controls
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        
        # Create worker
        self.worker = ClaudeWorker(self.claude)
        self.worker.response_received.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.progress_update.connect(self.update_status)
        
        # Send message
        self.worker.set_message(message, self.current_conversation_id)
        self.worker.start()
        
    def handle_response(self, response: str):
        """Handle Claude response"""
        self.add_message_to_display("assistant", response)
        
        # Re-enable controls
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.progress_bar.hide()
        
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #00cc00;")
        
    def handle_error(self, error_msg: str):
        """Handle errors"""
        QMessageBox.critical(self, "Error", f"Claude API Error: {error_msg}")
        
        # Re-enable controls
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.progress_bar.hide()
        
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff0000;")
        
    def update_status(self, status: str):
        """Update status message"""
        self.status_label.setText(status)
        
    def export_conversation(self):
        """Export conversation to file"""
        if not self.current_conversation_id:
            return
            
        # Get save path
        default_name = f"claude_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Conversation",
            default_name,
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if not filepath:
            return
            
        try:
            if filepath.endswith('.txt'):
                # Export as text
                conv = self.claude.conversations[self.current_conversation_id]
                with open(filepath, 'w') as f:
                    f.write(f"Claude Trading Analysis\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Created: {conv.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                    f.write(f"Context: {json.dumps(conv.context, indent=2)}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    for msg in conv.messages:
                        f.write(f"[{msg.timestamp.strftime('%H:%M:%S')}] ")
                        f.write(f"{'You' if msg.role == 'user' else 'Claude'}:\n")
                        f.write(f"{msg.content}\n")
                        f.write(f"{'-'*50}\n\n")
            else:
                # Export as JSON
                self.claude.export_conversation(self.current_conversation_id, filepath)
                
            QMessageBox.information(self, "Export Complete", f"Conversation exported to:\n{filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
            
    def new_conversation(self):
        """Start new conversation"""
        reply = QMessageBox.question(
            self,
            "New Conversation",
            "Start a new conversation? Current conversation will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.conversation_display.clear()
            self.current_conversation_id = self.claude.create_conversation(self.context)
            self.export_btn.setEnabled(False)
            self.status_label.setText("New conversation started")
            self.status_label.setStyleSheet("color: #00cc00;")


# ============= STANDALONE TEST =============
if __name__ == "__main__":
    print("=== Testing Claude Conversation Dialog ===\n")
    
    # Test data
    test_results = {
        'trend_1min': type('obj', (object,), {
            'signal': 'STRONG BUY',
            'confidence': 85.5,
            'strength': 78.2,
            'target_hold': '30-60 min'
        })(),
        'trend_5min': type('obj', (object,), {
            'direction': 'bullish',
            'strength': 82.3,
            'confidence': 90.1
        })()
    }
    
    test_context = {
        'symbol': 'NVDA',
        'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'timeframe': '1min'
    }
    
    app = QApplication(sys.argv)
    
    dialog = ClaudeConversationDialog(
        initial_results=test_results,
        context=test_context
    )
    
    dialog.show()
    
    sys.exit(app.exec())