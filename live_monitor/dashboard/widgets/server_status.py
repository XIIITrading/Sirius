# live_monitor/dashboard/widgets/server_status.py
"""
Server Status Widget - Visual indicator for server connection
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor


class ServerStatusWidget(QWidget):
    """Custom widget for server connection status indicator"""
    
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.setFixedSize(100, 30)
        
    def set_connected(self, connected: bool):
        """Update connection status"""
        self.is_connected = connected
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Custom paint for the indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Draw border
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        # Draw circle
        circle_color = QColor(68, 255, 68) if self.is_connected else QColor(255, 68, 68)
        painter.setBrush(QBrush(circle_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(10, 8, 14, 14)
        
        # Draw text
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(30, 20, "Server")