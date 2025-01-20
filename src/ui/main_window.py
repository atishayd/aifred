from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                            QLabel, QPushButton, QGridLayout, QLineEdit,
                            QFrame, QMessageBox, QHBoxLayout, QDialog, QScrollArea, QProgressBar, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from .course_dialog import CourseSelectionDialog
from .student_dialog import StudentDetailsDialog
from src.utils.logger import logger
import mediapipe as mp
import speech_recognition as sr
from openai import OpenAI
import threading
import queue
import pyaudio
import wave
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from math import ceil
import sip

class MainWindow(QMainWindow):
    # Only dark mode styles
    base_styles = {
        'dark': {
            'bg': '#2c2c2e',  # Consistent dark grey for all backgrounds
            'text': '#ffffff',
            'secondary_text': '#98989d',
            'accent': '#0A84FF',
            'accent_hover': '#0071e3',
            'border': '#48484a',
            'success': '#30d158',
            'error': '#ff453a',
        }
    }

    def get_style(self, theme='dark'):
        """Get theme-aware styles"""
        colors = self.base_styles[theme]
        return f"""
            /* Global Styles */
            QMainWindow, QDialog, QWidget, QTabWidget::pane {{
                background-color: {colors['bg']};
            }}
            
            /* Make sure all containers have the same background */
            QScrollArea, QScrollArea > QWidget > QWidget {{
                background-color: {colors['bg']};
            }}
            
            /* Student items and cards */
            .card {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding: 16px;
                margin: 8px;
            }}
            
            /* Ensure consistent backgrounds in lists */
            QListWidget, QListView {{
                background-color: {colors['bg']};
                border: none;
            }}
            
            /* Typography */
            QLabel {{
                color: {colors['text']};
                font-size: 14px;
                background-color: transparent;
            }}
            
            /* Inputs */
            QLineEdit {{
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 12px;
                color: {colors['text']};
                font-size: 14px;
                background-color: {colors['bg']};
            }}
            
            /* Rest of the styles... */
        """

    def __init__(self, db_manager, config, selected_course=None):
        super().__init__()
        self.db_manager = db_manager
        self.config = config
        
        # Allow resizing
        self.setMinimumSize(800, 600)
        
        # Initialize registration variables
        self.face_detection_attempts = 0
        self.max_detection_attempts = 5  # Maximum attempts to detect a face
        self.registration_camera_active = False
        self.captured_photo = None
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera variables
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        
        # Initialize face recognition data
        self.known_face_encodings = {}
        self.known_face_names = {}
        self.attendance_recorded = set()
        
        # Initialize hand raise tracking
        self.last_hand_raise = {}  # Track last hand raise time per student
        self.is_recording = False  # Track recording state
        
        # A helper method to center any QDialog on the screen
        def center_dialog_on_screen(dialog):
            # Make sure the dialog is sized first
            dialog.adjustSize()
            qr = dialog.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            dialog.move(qr.topLeft())

        # Only open the course selection if we did not receive a selected_course
        if selected_course is None:
            course_dialog = CourseSelectionDialog(self.db_manager, self)
            center_dialog_on_screen(course_dialog)
            if course_dialog.exec_() != QDialog.Accepted:
                raise Exception("No course selected")

            self.current_course = course_dialog.get_selected_course()
            if not self.current_course:
                raise Exception("No course selected")
        else:
            self.current_course = selected_course
        
        # Initialize course display variable
        course_name = self.current_course['course_name'].strip().upper()
        course_id = self.current_course['course_code'].strip()
        self.current_course_display = f"{course_name}{course_id}"  # Add this line
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Set window properties
        self.setWindowTitle("AiFRED - AI-powered Face Recognition Education System")
        
        # For a nicer user experience, let's maximize the main window
        self.showMaximized()
        
        # Initialize UI
        self.init_ui(layout)
        
        # Load face recognition data
        self.load_face_data()
        
        # Add analytics refresh timer
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.refresh_analytics)
        self.analytics_timer.start(5000)  # Refresh every 5 seconds

        # Store the index of the initial tab
        self.previous_tab_index = self.tab_widget.currentIndex()

    def init_ui(self, layout):
        """Initialize UI components"""
        # Create a scroll area for the entire content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 0px;
                background: transparent;
            }
        """)
        
        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a top bar using a QGridLayout so that the label stays centered
        top_bar_layout = QGridLayout()
        colors = self.base_styles['dark']

        # Format course display as COURSENAMEID
        course_name = self.current_course['course_name'].strip().upper()
        course_id = self.current_course['course_code'].strip()
        course_display = f"{course_name}{course_id}"

        course_info = QLabel(course_display)
        course_info.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            padding: 16px;
            color: {colors['accent']};
            letter-spacing: 1px;
        """)
        course_info.setAlignment(Qt.AlignCenter)
        course_info.setCursor(Qt.PointingHandCursor)
        course_info.mousePressEvent = lambda e: self.show_course_details()

        exit_button = QPushButton("Exit")
        exit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['error']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                margin-right: 16px;
            }}
            QPushButton:hover {{
                background-color: #ff2d20;
            }}
        """)
        exit_button.clicked.connect(self.exit_to_menu)

        # Set column stretch so column 0 and column 2 expand equally,
        # keeping the label (column 1) centered while exit is pinned right (also col 2).
        top_bar_layout.setColumnStretch(0, 1)  # Expand left side
        top_bar_layout.setColumnStretch(2, 1)  # Expand to the right
        # Add the course_info in the middle column, aligned center
        top_bar_layout.addWidget(course_info, 0, 1, alignment=Qt.AlignCenter)
        # Place exit_button in the rightmost column, aligned to the right
        top_bar_layout.addWidget(exit_button, 0, 2, alignment=Qt.AlignRight)

        container_layout.addLayout(top_bar_layout)
        
        # Create tab widget with dark styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: transparent;
                margin-top: -1px;
            }}
            QTabBar::tab {{
                min-width: 150px;
                padding: 12px 24px;
                margin: 0 4px;
                border-bottom: 2px solid transparent;
                font-weight: 500;
                font-size: 14px;
                color: {colors['secondary_text']};
            }}
            QTabBar {{
                alignment: center;
            }}
            QTabBar::tab:hover:!selected {{
                color: #ffffff;
            }}
            QTabBar::tab:selected {{
                color: #ffffff;
                border-bottom: 2px solid {colors['accent']};
            }}
        """)
        
        # Create and add tabs with descriptive but concise names
        students_tab = QWidget()
        attendance_tab = QWidget()
        engagement_tab = QWidget()
        analytics_tab = QWidget()
        registration_tab = QWidget()
        
        # Give the attendance tab an object name so we can detect it later
        attendance_tab.setObjectName("attendance_tab")  
        registration_tab.setObjectName("registration_tab")
        
        self.tab_widget.addTab(students_tab, "Students")
        self.tab_widget.addTab(attendance_tab, "Attendance")
        self.tab_widget.addTab(engagement_tab, "Engagement")
        self.tab_widget.addTab(analytics_tab, "Analytics")
        self.tab_widget.addTab(registration_tab, "Registration")
        
        # Set up tab contents
        self._setup_students_tab(students_tab)
        self._setup_attendance_tab(attendance_tab)
        self._setup_engagement_tab(engagement_tab)
        self._setup_analytics_tab(analytics_tab)
        self._setup_registration_tab(registration_tab)
        
        # Connect signal using self.tab_widget instead of tabs
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        container_layout.addWidget(self.tab_widget)
        
        # Set scroll area widget
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

    def _setup_students_tab(self, tab):
        """Setup the students panel"""
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)  # Add some padding
        
        # Get colors
        colors = self.base_styles['dark']
        
        # Add search controls with improved styling
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        
        # Style the search label
        search_label = QLabel("Search:")
        search_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                font-size: 14px;
                font-weight: 500;
            }}
        """)
        
        # Create and style search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search students...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 12px;
                color: {colors['text']};
                font-size: 14px;
                background-color: {colors['bg']};
            }}
            QLineEdit::placeholder {{
                color: {colors['secondary_text']};
            }}
            QLineEdit:focus {{
                border: 2px solid {colors['accent']};
            }}
        """)
        
        # Connect search input to filter function
        self.search_input.textChanged.connect(self.filter_students)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Create list for students with hidden scrollbar
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {colors['bg']};
            }}
            QScrollBar:vertical {{
                width: 0px;
            }}
        """)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {colors['bg']};
            }}
        """)
        
        self.students_list = QVBoxLayout(scroll_widget)
        self.students_list.setSpacing(10)
        self.students_list.setAlignment(Qt.AlignTop)
        self.students_list.setContentsMargins(0, 10, 0, 10)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        self.update_student_list()

    def filter_students(self):
        """Filter students based on search text"""
        search_text = self.search_input.text().lower()
        
        # Get all students
        students = self.db_manager.get_course_students(self.current_course['_id'])
        
        # Clear existing items
        while self.students_list.count():
            item = self.students_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add filtered student items
        for student in students:
            if search_text in student['name'].lower():
                item = self.create_student_item(student)
                self.students_list.addWidget(item)

    def create_student_item(self, student):
        """Create a modern, card-like student item"""
        item = QWidget()
        colors = self.base_styles['dark']
        
        # Container with no margin or border
        item.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                border: none;
                margin: 0px;
            }}
        """)
        
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Container for all content with single border
        content = QWidget()
        content.setStyleSheet(f"""
            QWidget {{
                background-color: {colors['bg']};
                border: none;  /* Removed border */
                padding: 16px;
            }}
        """)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)
        
        # Name label with reactive color
        name = QLabel(student['name'])
        name.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 500;
            color: {colors['text']};
        """)
        
        # Add widgets to content layout
        content_layout.addWidget(name, stretch=1)
        
        # Actions container
        actions = QHBoxLayout()
        actions.setSpacing(8)
        
        details_btn = QPushButton("Details")
        details_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['border']};
                color: {colors['text']};
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {colors['secondary_text']};
                color: {colors['bg']};
            }}
        """)
        details_btn.clicked.connect(lambda: self.show_student_details(student))
        
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['error']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #ff2d20;
            }}
        """)
        remove_btn.clicked.connect(lambda: self.remove_student(student))
        
        actions.addWidget(details_btn)
        actions.addWidget(remove_btn)
        content_layout.addLayout(actions)
        
        # Add content widget to main layout
        layout.addWidget(content)
        
        return item

    def update_student_list(self):
        """Update the list of students"""
        # Clear existing items
        while self.students_list.count():
            item = self.students_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get students for current course
        students = self.db_manager.get_course_students(self.current_course['_id'])
        
        # Add student items to list
        for student in students:
            item = self.create_student_item(student)
            self.students_list.addWidget(item)

    def show_student_details(self, student):
        """Show student details in a theme-aware dialog"""
        colors = self.base_styles['dark']
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Student Details")
        dialog.setMinimumWidth(400)
        
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['bg']};
                padding: 20px;
            }}
            QLabel {{
                color: {colors['text']};
            }}
            QPushButton {{
                background-color: #e0e0e0;
                color: #2c2c2e;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
                font-size: 13px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background-color: {colors['accent']};
                color: white;
            }}
            QLineEdit {{
                background-color: {colors['bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        
        # Name section
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setStyleSheet("font-weight: bold;")
        name_value = QLabel(student['name'])
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_value)
        layout.addLayout(name_layout)
        
        # ID section
        id_layout = QHBoxLayout()
        id_label = QLabel("ID:")
        id_label.setStyleSheet("font-weight: bold;")
        id_value = QLabel(str(student['student_id']))
        id_layout.addWidget(id_label)
        id_layout.addWidget(id_value)
        layout.addLayout(id_layout)
        
        # Photo section
        photo_label = QLabel()
        photo_label.setMinimumSize(300, 300)
        photo_label.setAlignment(Qt.AlignCenter)
        
        try:
            # Load and process photo
            photo_array = np.frombuffer(student['photo'], dtype=np.uint8)
            decoded_image = cv2.imdecode(photo_array, cv2.IMREAD_COLOR)
            if decoded_image is not None:
                # Convert to RGB without drawing face detection box
                rgb_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                photo_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logger.error(f"Error loading student photo: {str(e)}")
            photo_label.setText("No Photo Available")
        
        layout.addWidget(photo_label)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        rename_btn = QPushButton("Rename")
        close_btn = QPushButton("Close")
        
        def handle_rename():
            # Create rename dialog
            rename_dialog = QDialog(dialog)
            rename_dialog.setWindowTitle("Rename Student")
            rename_dialog.setStyleSheet(dialog.styleSheet())
            
            rename_layout = QVBoxLayout(rename_dialog)
            
            # Add input field
            new_name_input = QLineEdit()
            new_name_input.setText(student['name'])
            new_name_input.setPlaceholderText("Enter new name...")
            rename_layout.addWidget(new_name_input)
            
            # Add buttons
            rename_buttons = QHBoxLayout()
            save_btn = QPushButton("Save")
            cancel_btn = QPushButton("Cancel")
            
            def save_new_name():
                new_name = new_name_input.text().strip()
                if new_name:
                    try:
                        # Update in database
                        self.db_manager.update_student_name(
                            student_id=student['student_id'],
                            course_id=self.current_course['_id'],
                            new_name=new_name
                        )
                        
                        # Update UI
                        name_value.setText(new_name)
                        self.update_student_list()
                        self.update_engagement_list()
                        
                        # Show success message
                        msg = self.create_styled_message_box(
                            QMessageBox.Information,
                            "Success",
                            "Student name updated successfully"
                        )
                        msg.exec_()
                        rename_dialog.accept()
                        
                    except Exception as e:
                        error_msg = self.create_styled_message_box(
                            QMessageBox.Critical,
                            "Error",
                            "Failed to update student name",
                            str(e)
                        )
                        error_msg.exec_()
                else:
                    error_msg = self.create_styled_message_box(
                        QMessageBox.Warning,
                        "Invalid Name",
                        "Please enter a valid name"
                    )
                    error_msg.exec_()
            
            save_btn.clicked.connect(save_new_name)
            cancel_btn.clicked.connect(rename_dialog.reject)
            
            rename_buttons.addWidget(save_btn)
            rename_buttons.addWidget(cancel_btn)
            rename_layout.addLayout(rename_buttons)
            
            rename_dialog.exec_()
        
        rename_btn.clicked.connect(handle_rename)
        close_btn.clicked.connect(dialog.accept)
        
        buttons_layout.addWidget(rename_btn)
        buttons_layout.addWidget(close_btn)
        layout.addLayout(buttons_layout)
        
        dialog.exec_()

    def remove_student(self, student):
        """Remove a student with theme-aware confirmation dialog"""
        try:
            colors = self.base_styles['dark']
            
            # Create custom confirmation dialog
            confirm = QMessageBox(self)
            confirm.setWindowTitle("Remove Student")
            confirm.setText("Remove student?")
            confirm.setInformativeText("This will remove all attendance, engagement\nrecords, and questions for this student.")
            confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm.setDefaultButton(QMessageBox.No)
            confirm.setIcon(QMessageBox.NoIcon)  # Remove icon
            
            # Apply improved styling
            confirm.setStyleSheet(f"""
                QMessageBox {{
                    background-color: {colors['bg']};
                    min-width: 300px;
                    max-width: 300px;
                }}
                QMessageBox QLabel {{
                    color: {colors['text']};
                    font-size: 14px;
                    padding: 10px;
                    alignment: center;
                }}
                QMessageBox QLabel#qt_msgbox_label {{
                    color: {colors['text']};
                    font-size: 15px;
                    font-weight: 500;
                    padding-top: 20px;
                }}
                QMessageBox QLabel#qt_msgbox_informativelabel {{
                    color: {colors['secondary_text']};
                    font-size: 13px;
                    padding-bottom: 20px;
                }}
                QMessageBox QPushButton {{
                    background-color: #e0e0e0;
                    color: #2c2c2e;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    min-width: 80px;
                    font-size: 13px;
                    margin: 0px 5px;
                }}
                QPushButton[text="Yes"]:hover {{
                    background-color: {colors['error']};
                    color: white;
                }}
                QPushButton[text="No"]:hover {{
                    background-color: {colors['accent']};
                    color: white;
                }}
            """)
            
            # Center align all labels
            for label in confirm.findChildren(QLabel):
                label.setAlignment(Qt.AlignCenter)
            
            if confirm.exec_() == QMessageBox.Yes:
                # Remove student
                result = self.db_manager.remove_student(
                    student_id=student['student_id'],
                    course_id=self.current_course['_id']
                )
                
                if result:
                    # Show success message
                    success = self.create_styled_message_box(
                        QMessageBox.Information,
                        "Student Removed",
                        f"{student['name']} has been removed",
                        "All records have been deleted."
                    )
                    success.exec_()
                    
                    # Update all relevant UI components
                    self.update_student_list()
                    self.update_engagement_list()
                    self.load_face_data()
                    self.refresh_analytics()  # Refresh analytics immediately
                    
                else:
                    raise Exception("Failed to remove student from database")
                
        except Exception as e:
            error = self.create_styled_message_box(
                QMessageBox.Critical,
                "Error",
                "Failed to remove student",
                str(e)
            )
            error.exec_()

    def _setup_attendance_tab(self, tab):
        """
        Setup the attendance tracking tab with user-friendly interface.
        Allow indefinite recording so multiple students can ask questions
        across a longer session.
        """
        colors = self.base_styles['dark']

        layout = QVBoxLayout(tab)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.attendance_welcome_container = QWidget()
        self.attendance_welcome_container.setStyleSheet("QWidget { background: transparent; }")
        welcome_layout = QVBoxLayout(self.attendance_welcome_container)
        welcome_layout.setSpacing(20)

        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setSpacing(20)
        center_layout.setAlignment(Qt.AlignCenter)

        # Renamed the title and explanatory text
        welcome_title = QLabel("Class Attendance & Questions")
        welcome_title.setStyleSheet(f"color: {colors['text']}; font-size: 28px; font-weight: 600;")
        welcome_title.setAlignment(Qt.AlignCenter)

        welcome_text = QLabel("Press 'Start Class Recording' to begin a class session.")
        welcome_text.setStyleSheet(f"color: {colors['text']}; font-size: 16px; qproperty-wordWrap: true;")
        welcome_text.setAlignment(Qt.AlignCenter)

        # 'Start Class Recording' button
        self.start_recording_button = QPushButton("Start Class Recording")
        self.start_recording_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: 500;
                min-width: 200px;
            }}
            QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
        """)
        self.start_recording_button.clicked.connect(self.start_class_recording)

        center_layout.addStretch()
        center_layout.addWidget(welcome_title)
        center_layout.addWidget(welcome_text)
        center_layout.addWidget(self.start_recording_button)
        center_layout.addStretch()

        welcome_layout.addWidget(center_container)

        # Attendance container (camera + stop button)
        self.attendance_container = QWidget()
        self.attendance_container.hide()
        attendance_layout = QVBoxLayout(self.attendance_container)

        self.attendance_status = QLabel()
        self.attendance_status.setStyleSheet(f"color: {colors['text']}; font-size: 20px; font-weight: 500; margin: 20px;")
        self.attendance_status.setAlignment(Qt.AlignCenter)
        attendance_layout.addWidget(self.attendance_status)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""QLabel { background-color: transparent; border: none; }""")
        attendance_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['error']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: #ff2d20;
            }}
            QPushButton:disabled {{
                background-color: {colors['border']};
                color: {colors['secondary_text']};
            }}
        """)
        # Stop button ends the indefinite session
        self.stop_button.clicked.connect(self.stop_camera)
        attendance_layout.addWidget(self.stop_button)

        layout.addWidget(self.attendance_welcome_container)
        layout.addWidget(self.attendance_container)

    def start_class_recording(self):
        """
        Start an indefinite class recording session.
        Students can raise their hands and ask multiple questions.
        """
        self.is_class_recording = True  # Flag to denote "class is in session"

        # Hide the welcome container, show the attendance container
        self.attendance_welcome_container.hide()
        self.attendance_container.show()
        self.attendance_status.setText("Recording in progress. Multiple students can ask questions...")

        # Start camera feed (if not already started)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            msg = self.create_styled_message_box(
                QMessageBox.Critical,
                "Error",
                "Could not access the camera."
            )
            msg.exec_()
            return

        # Start updating camera feed
        self.camera_timer.start(30)  # ~30 fps
        self.start_recording_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Clear existing attendance for a new session if you prefer:
        # self.attendance_recorded.clear()

    def stop_camera(self):
        """
        Stop camera feed and end the class session.
        """
        self.is_class_recording = False  # No longer recording the class session

        # Stop camera and timer
        if hasattr(self, 'camera') and self.camera:
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None

        self.camera_label.clear()

        self.start_recording_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Hide attendance container and show welcome screen
        self.attendance_container.hide()
        self.attendance_welcome_container.show()
        self.attendance_status.setText("")
        # Optionally clear the attendance_recorded set if desired:
        # self.attendance_recorded.clear()

    def update_camera_feed(self):
        """Update camera feed for registration or attendance"""
        if not hasattr(self, 'camera') or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Convert frame to RGB for face recognition *and* mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if hasattr(self, 'registration_camera_active') and self.registration_camera_active:
            self.handle_registration_feed(frame, rgb_frame)
        else:
            # Attendance camera feed
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                matches = []
                for student_id, known_encoding in self.known_face_encodings.items():
                    if face_recognition.compare_faces([known_encoding], face_encoding)[0]:
                        matches.append(student_id)
                
                # Record attendance if student wasn't already recorded
                if matches and matches[0] not in self.attendance_recorded:
                    student_id = matches[0]
                    name = self.known_face_names.get(student_id, "Unknown")
                    self.record_attendance(student_id, name)

                # Even if attendance is recorded, keep calling pose detection for ongoing tracking
                if matches:
                    student_id = matches[0]
                    name = self.known_face_names.get(student_id, "Unknown")
                    # Call pose detection every frame for recognized faces
                    self.process_student_actions(frame, rgb_frame, student_id, name)
        
        # Convert BGR frame back to RGB for display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_student_actions(self, frame_bgr, frame_rgb, student_id, name):
        """Process pose detection (in RGB) and draw results on the BGR frame."""
        # Get image dimensions
        height, width = frame_rgb.shape[:2]
        
        # Configure MediaPipe with image dimensions
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Pose detection must use an RGB image
        pose_results = self.pose.process(frame_rgb)
        
        # If pose is found, do hand-raise detection
        if pose_results.pose_landmarks:
            self.check_hand_raise(student_id, name, pose_results.pose_landmarks.landmark)
            
            # Draw the pose landmarks on the BGR frame
            self.mp_draw.draw_landmarks(
                frame_bgr,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

    def record_attendance(self, student_id, student_name):
        """Record attendance for a student"""
        try:
            # Mark attendance in database
            self.db_manager.mark_attendance(
                student_id=student_id,
                date_param=datetime.now(),  # Changed from 'date' to 'date_param'
                status='Present',
                course_id=self.current_course['_id']
            )
            
            # Update UI to show attendance recorded
            self.attendance_status.setText(f"Attendance recorded for {student_name}")
            self.attendance_recorded.add(student_id)
            
        except Exception as e:
            logger.error(f"Error recording attendance for {student_name}: {str(e)}")

    def show_attendance_notification(self, name):
        """Show a temporary notification for attendance"""
        msg = self.create_styled_message_box(
            QMessageBox.Information,
            "Attendance Recorded",
            f"Attendance recorded for {name}"
        )
        QTimer.singleShot(2000, msg.accept)
        msg.exec_()

    def closeEvent(self, event):
        """Clean up resources when closing the window"""
        self.stop_camera()
        self.stop_registration_camera()
        event.accept()

    def _setup_engagement_tab(self, tab):
        """Setup the engagement tracking tab"""
        # Set object name for tab identification
        tab.setObjectName("engagement_tab")
        
        # Create scroll area
        scroll_area = QScrollArea(tab)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 0px;
                background: transparent;
            }
        """)
        
        # Create container widget
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background: transparent;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create engagement list container
        engagement_container = QWidget()
        engagement_container.setStyleSheet("""
            QWidget {
                background: transparent;
            }
        """)
        self.engagement_list = QVBoxLayout(engagement_container)
        self.engagement_list.setSpacing(5)
        self.engagement_list.setAlignment(Qt.AlignTop)
        layout.addWidget(engagement_container)
        
        # Set the container as the scroll area widget
        scroll_area.setWidget(container)
        
        # Add scroll area to tab layout
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)

    def update_engagement_list(self):
        """Update the engagement list with clean styling"""
        try:
            logger.info("Updating engagement list...")
            colors = self.base_styles['dark']
            
            # Clear existing items
            for i in reversed(range(self.engagement_list.count())):
                self.engagement_list.itemAt(i).widget().deleteLater()
            
            # Get students in current course
            students = self.db_manager.get_course_students(self.current_course['_id'])
            
            # Create a mapping of student IDs to names
            student_names = {str(student['student_id']): student['name'] for student in students}
            
            for student in students:
                questions = self.db_manager.get_student_questions(
                    student['student_id'], 
                    self.current_course['_id']
                )
                
                if questions:
                    container = QWidget()
                    container.setStyleSheet(f"""
                        QWidget {{
                            background-color: transparent;
                            padding: 0px;
                            margin: 0px;
                        }}
                    """)
                    
                    container_layout = QVBoxLayout(container)
                    container_layout.setSpacing(0)
                    container_layout.setContentsMargins(0, 0, 0, 0)
                    
                    for i, q in enumerate(questions):
                        # Add student name to question data
                        q['student_name'] = student_names.get(str(q['student_id']), 'Unknown Student')
                        q['is_last'] = (i == len(questions) - 1)
                        q_widget = self.create_question_card(q)
                        container_layout.addWidget(q_widget)
                    
                    self.engagement_list.addWidget(container)
            
            logger.info("Engagement list updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating engagement list: {str(e)}")

    def create_engagement_item(self, student):
        """Create an engagement list item with modern card design"""
        item = QWidget()
        item.setStyleSheet(f"""
            QWidget {{
                background: {('#2c2c2e')};
                border-radius: 12px;
                margin: 8px;
                padding: 16px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, {('0.3')});
            }}
            QWidget:hover {{
                box-shadow: 0 4px 12px rgba(0, 0, 0, {('0.4')});
                transform: translateY(-2px);
                transition: all 0.3s ease;
            }}
        """)
        
        item_layout = QVBoxLayout(item)
        item_layout.setSpacing(12)
        item_layout.setContentsMargins(16, 16, 16, 16)
        
        # Student name with avatar-like circle
        header_layout = QHBoxLayout()
        avatar_label = QLabel()
        avatar_label.setFixedSize(40, 40)
        avatar_label.setStyleSheet("""
            background: #007AFF;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 16px;
        """)
        avatar_label.setText(student['name'][0].upper())
        avatar_label.setAlignment(Qt.AlignCenter)
        
        name_label = QLabel(student['name'])
        name_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #1d1d1f;
        """)
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(name_label)
        header_layout.addStretch()
        
        item_layout.addLayout(header_layout)
        
        return item

    def show_student_engagement(self, student):
        """Show student engagement details"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Engagement Details - {student['name']}")
        layout = QVBoxLayout(dialog)
        
        # Create scroll area for questions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        def update_questions():
            # Clear existing questions
            while scroll_layout.count():
                item = scroll_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Get updated questions
            questions = self.db_manager.get_student_questions(
                student['student_id'],
                self.current_course['_id']
            )
            
            # Add question cards
            for question in questions:
                card = self.create_question_card(question)
                scroll_layout.addWidget(card)
        
        # Initial update
        update_questions()
        
        # Set up timer for periodic updates
        update_timer = QTimer(dialog)
        update_timer.timeout.connect(update_questions)
        update_timer.start(5000)  # Update every 5 seconds
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def _setup_analytics_tab(self, tab):
        """Setup the analytics tab with improved visualization and layout"""
        # Set object name for tab identification
        tab.setObjectName("analytics_tab")
        
        # Create scroll area
        scroll_area = QScrollArea(tab)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 0px;
                background: transparent;
            }
        """)
        
        # Create container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Populate the layout
        self._populate_analytics_tab(layout)
        
        # Set the container as the scroll area widget
        scroll_area.setWidget(container)
        
        # Add scroll area to tab layout
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)

    def _populate_analytics_tab(self, layout):
        """Populate the analytics tab layout with content"""
        colors = self.base_styles['dark']
        
        # Create a scroll area to contain everything
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        # Create main container widget
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setSpacing(30)
        main_layout.setAlignment(Qt.AlignTop)
        
        # 1. Top Students Section
        top_students_title = QLabel("Top Students")
        top_students_title.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 600;
                padding-bottom: 20px;
            }}
        """)
        top_students_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(top_students_title)

        # Create container for student sections with fixed width
        students_container = QWidget()
        students_container.setFixedWidth(1000)  # Set fixed width for consistency
        students_layout = QHBoxLayout(students_container)
        students_layout.setSpacing(40)
        students_layout.setAlignment(Qt.AlignCenter)

        # Get student data and create sections
        questions = self.db_manager.get_course_questions(self.current_course['_id'])
        student_questions = {}
        for q in questions:
            student_id = q['student_id']
            if student_id not in student_questions:
                student_questions[student_id] = {'relevant': 0, 'irrelevant': 0}
            if q.get('is_relevant', True):
                student_questions[student_id]['relevant'] += 1
            else:
                student_questions[student_id]['irrelevant'] += 1

        # Create sections
        relevant_section = self._create_student_section(
            "Most Relevant Questions",
            sorted(student_questions.items(), key=lambda x: x[1]['relevant'], reverse=True)[:3],
            colors,
            metric='relevant'
        )
        students_layout.addWidget(relevant_section)

        irrelevant_section = self._create_student_section(
            "Most Irrelevant Questions",
            sorted(student_questions.items(), key=lambda x: x[1]['irrelevant'], reverse=True)[:3],
            colors,
            metric='irrelevant'
        )
        students_layout.addWidget(irrelevant_section)
        main_layout.addWidget(students_container, alignment=Qt.AlignCenter)

        # 2. Attendance Section
        attendance_title = QLabel(f"Attendance for {datetime.now().strftime('%d %B')}")
        attendance_title.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 600;
                padding: 20px 0px;
            }}
        """)
        attendance_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(attendance_title)

        # Get attendance data
        today = datetime.now().date()
        students = self.db_manager.get_course_students(self.current_course['_id'])
        attendance_records = self.db_manager.get_attendance_records(
            course_id=self.current_course['_id'],
            date_param=today
        )

        # Create attendance list with fixed width
        attendance_list = QWidget()
        attendance_list.setFixedWidth(400)
        list_layout = QVBoxLayout(attendance_list)
        list_layout.setSpacing(10)
        
        present_students = {record['student_id'] for record in attendance_records if record['status'] == 'Present'}
        
        for student in students:
            student_container = QWidget()
            student_layout = QHBoxLayout(student_container)
            student_layout.setContentsMargins(0, 0, 0, 0)
            
            name_label = QLabel(student['name'])
            name_label.setStyleSheet(f"""
                color: {colors['text']};
                font-size: 16px;
                font-weight: 500;
            """)
            
            is_present = student['student_id'] in present_students
            status_label = QLabel("Present" if is_present else "Absent")
            status_label.setStyleSheet(f"""
                color: {colors['success'] if is_present else colors['error']};
                font-size: 16px;
                font-weight: 600;
                padding: 4px 12px;
                border-radius: 4px;
            """)
            
            student_layout.addWidget(name_label)
            student_layout.addStretch()
            student_layout.addWidget(status_label)
            
            list_layout.addWidget(student_container)
        
        main_layout.addWidget(attendance_list, alignment=Qt.AlignCenter)

        # 3. Charts Section
        charts_title = QLabel("Analytics")
        charts_title.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 600;
                padding: 20px 0px;
            }}
        """)
        charts_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(charts_title)

        # Create container for both charts with fixed width
        charts_container = QWidget()
        charts_container.setFixedWidth(1000)  # Set fixed width for consistency
        charts_layout = QHBoxLayout(charts_container)
        charts_layout.setSpacing(40)
        charts_layout.setAlignment(Qt.AlignCenter)

        # Left side: Pie Chart
        pie_chart_container = QWidget()
        pie_layout = QVBoxLayout(pie_chart_container)
        pie_layout.setAlignment(Qt.AlignCenter)

        pie_title = QLabel("Question Relevance Distribution")
        pie_title.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 16px;
            font-weight: 600;
            padding-bottom: 10px;
        """)
        pie_title.setAlignment(Qt.AlignCenter)
        pie_layout.addWidget(pie_title)

        total_questions = len(questions)
        relevant_count = sum(1 for q in questions if q.get('is_relevant', True))
        irrelevant_count = total_questions - relevant_count

        pie_chart_canvas = self.create_pie_chart(relevant_count, irrelevant_count)
        pie_layout.addWidget(pie_chart_canvas, alignment=Qt.AlignCenter)
        charts_layout.addWidget(pie_chart_container)

        # Right side: Hand Raises Bar Graph
        bar_chart_container = QWidget()
        bar_layout = QVBoxLayout(bar_chart_container)
        bar_layout.setAlignment(Qt.AlignCenter)

        bar_title = QLabel("Hand Raises per Student")
        bar_title.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 16px;
            font-weight: 600;
            padding-bottom: 10px;
        """)
        bar_title.setAlignment(Qt.AlignCenter)
        bar_layout.addWidget(bar_title)

        bar_chart_canvas = self.create_hand_raises_chart(students)
        bar_layout.addWidget(bar_chart_canvas, alignment=Qt.AlignCenter)
        charts_layout.addWidget(bar_chart_container)

        main_layout.addWidget(charts_container, alignment=Qt.AlignCenter)

        # Set the main container as the scroll area's widget
        scroll_area.setWidget(main_container)
        
        # Add scroll area to the main layout
        layout.addWidget(scroll_area)

    def _create_student_section(self, title, students, colors, width=500, metric='relevant'):
        """Create a styled section for student lists"""
        section = QWidget()
        section.setFixedWidth(width)
        section.setStyleSheet("""
            QWidget {
                background-color: transparent;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setSpacing(20)
        
        # Add title with colored background based on type
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(title)
        if metric == 'relevant':
            bg_color = colors['success']
        else:
            bg_color = colors['error']
        
        title_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                font-size: 20px;
                font-weight: 600;
                padding: 12px 24px;
                background-color: {bg_color};
                border-radius: 12px;
                margin: 0px 10px;
            }}
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        layout.addWidget(title_container)
        
        # Add students with larger text
        for student_id, counts in students:
            # Get student name from database
            student = self.db_manager.get_student_by_id(student_id)
            if student:
                student_name = student.get('name', 'Unknown Student')
                count = counts[metric]  # Get the relevant or irrelevant count
                
                student_widget = QWidget()
                student_layout = QHBoxLayout(student_widget)
                student_layout.setAlignment(Qt.AlignCenter)
                
                name_label = QLabel(student_name)  # Use student name instead of ID
                name_label.setStyleSheet(f"""
                    color: {colors['text']};
                    font-size: 18px;
                    font-weight: 500;
                """)
                name_label.setAlignment(Qt.AlignCenter)
                
                count_label = QLabel(str(count))
                count_label.setStyleSheet("""
                    color: white;
                    padding: 5px 10px;
                    font-weight: bold;
                    font-size: 18px;
                """)
                count_label.setAlignment(Qt.AlignCenter)
                
                student_layout.addWidget(name_label)
                student_layout.addWidget(count_label)
                layout.addWidget(student_widget)
        
        # Add bottom padding to ensure nothing is cut off
        spacer = QWidget()
        spacer.setFixedHeight(20)
        layout.addWidget(spacer)
        
        return section

    def _create_attendance_list(self, colors):
        """Create a styled attendance list"""
        attendance_widget = QWidget()
        layout = QVBoxLayout(attendance_widget)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)  # Center the entire list
        
        # Get attendance data
        students = self.db_manager.get_course_students(self.current_course['_id'])
        
        for student in students:
            student_row = QWidget()
            row_layout = QHBoxLayout(student_row)
            row_layout.setSpacing(20)
            row_layout.setAlignment(Qt.AlignCenter)  # Center the row contents
            
            # Name
            name_label = QLabel(student['name'])
            name_label.setStyleSheet(f"color: {colors['text']};")
            name_label.setAlignment(Qt.AlignCenter)  # Center the name
            
            # Status
            status = "Present" if student.get('is_present', False) else "Absent"
            status_label = QLabel(status)
            status_color = colors['success'] if status == "Present" else colors['error']
            status_label.setStyleSheet(f"""
            color: white;
                background-color: {status_color};
                border-radius: 10px;
                padding: 5px 15px;
            """)
            status_label.setAlignment(Qt.AlignCenter)  # Center the status
            
            # Add widgets to layout
            row_layout.addWidget(name_label)
            row_layout.addWidget(status_label)
            
            layout.addWidget(student_row)
        
        return attendance_widget

    def _setup_registration_tab(self, tab):
        """Setup the registration tab"""
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Create container for welcome screen
        self.welcome_container = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_container)
        welcome_layout.setAlignment(Qt.AlignCenter)  # Center everything vertically
        
        # Get colors
        colors = self.base_styles['dark']
        
        # Welcome title
        welcome_title = QLabel("Student Registration")
        welcome_title.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
        """)
        welcome_title.setAlignment(Qt.AlignCenter)
        
        # Welcome text
        welcome_text = QLabel("Ready to register a new student? We'll guide you through the process.")
        welcome_text.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 16px;
            margin-bottom: 30px;
            qproperty-wordWrap: true;
        """)
        welcome_text.setAlignment(Qt.AlignCenter)
        welcome_text.setFixedWidth(400)
        
        # Ready button
        self.ready_btn = QPushButton("I'm Ready!")
        self.ready_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: 500;
                min-width: 200px;
            }}
            QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
        """)
        self.ready_btn.clicked.connect(self.start_registration_process)
        
        # Add elements to welcome layout
        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(welcome_text)
        welcome_layout.addWidget(self.ready_btn)
        welcome_layout.addStretch()
        
        # Create container for registration process
        self.registration_container = QWidget()
        registration_layout = QVBoxLayout(self.registration_container)
        registration_layout.setAlignment(Qt.AlignCenter)  # Center everything vertically
        
        # Style the camera feed without text or border
        self.camera_feed = QLabel()  # Remove the "Camera Preview" text
        self.camera_feed.setFixedSize(640, 480)
        self.camera_feed.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
            }
        """)
        self.camera_feed.setAlignment(Qt.AlignCenter)
        registration_layout.addWidget(self.camera_feed)
        
        # Add status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 16px;
            margin: 10px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        registration_layout.addWidget(self.status_label)
        
        # Add name input (initially hidden)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter student name...")
        self.name_input.hide()
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {colors['bg']};
                color: {colors['text']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                margin-top: 20px;
            }}
        """)
        registration_layout.addWidget(self.name_input)
        
        # Add register button (initially hidden)
        self.register_btn = QPushButton("Register Student")
        self.register_btn.hide()
        self.register_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
            }}
        """)
        self.register_btn.clicked.connect(self.register_student)
        registration_layout.addWidget(self.register_btn)
        
        # Add containers to main layout
        layout.addWidget(self.welcome_container)
        layout.addWidget(self.registration_container)
        
        # Initially hide registration container
        self.registration_container.hide()

    def start_registration_process(self):
        """Start the guided registration process"""
        # If coming from welcome screen, hide it
        if self.welcome_container.isVisible():
            self.welcome_container.hide()
            self.registration_container.show()
        
        # Ensure any existing timer is cleaned up
        if hasattr(self, 'timer_widget') and self.timer_widget:
            try:
                self.timer_widget.hide()
                self.timer_widget.deleteLater()
            except RuntimeError:
                pass  # Widget already deleted
            self.timer_widget = None
        
        if hasattr(self, 'countdown_timer') and self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        
        # Create and setup circular timer
        self.timer_widget = CircularTimer(self)
        self.timer_widget.move(
            (self.registration_container.width() - self.timer_widget.width()) // 2,
            (self.registration_container.height() - self.timer_widget.height()) // 2
        )
        
        # Start countdown
        self.time_left = 3.0
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(33)  # ~30fps for smooth animation
        
        self.timer_widget.show()

    def update_countdown(self):
        """Update the countdown timer"""
        if not hasattr(self, 'time_left'):
            return
            
        self.time_left = max(0, self.time_left - 0.033)  # Decrease by 33ms
        
        # Update circular timer
        progress = (3.0 - self.time_left) / 3.0  # 0 to 1
        self.timer_widget.value = int(progress * 360)  # Convert to degrees
        self.timer_widget.number = max(1, int(ceil(self.time_left)))
        self.timer_widget.update()  # Trigger repaint
        
        if self.time_left <= 0:
            self.countdown_timer.stop()
            self.timer_widget.hide()
            self.timer_widget.deleteLater()
            self.status_label.setText("Looking for face...")
            self.start_camera_detection()

    def start_camera_detection(self):
        """Start camera and look for face"""
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.registration_camera_active = True
            self.camera_timer.start(30)
            # Initialize face detection attempts counter
            self.face_detection_attempts = 0  # Initialize to 0
            self.max_detection_attempts = 100  # About 3 seconds at 30fps
        else:
            self.status_label.setText("Error: Could not access camera")

    def handle_registration_feed(self, frame, rgb_frame):
        """Handle camera feed for registration with automatic capture"""
        # Initialize counter if not exists
        if not hasattr(self, 'face_detection_attempts'):
            self.face_detection_attempts = 0
            self.max_detection_attempts = 100
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw rectangles around faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        if len(face_locations) == 1:
            self.status_label.setText("Face detected! Capturing in 1 second...")
            QTimer.singleShot(1000, lambda: self.auto_capture_photo(frame))
            self.face_detection_attempts = None  # Stop counting attempts
        else:
            # Only increment if attempts is not None
            if self.face_detection_attempts is not None:
                self.face_detection_attempts += 1
                if self.face_detection_attempts >= self.max_detection_attempts:
                    # First stop the camera
                    self.stop_registration_camera()
                    
                    # Show error message and connect to restart process
                    msg = self.create_styled_message_box(
                        QMessageBox.Critical,
                        "Failed to register student",
                        "No face detected in photo. Please try again."
                    )
                    msg.buttonClicked.connect(self.handle_restart)  # Connect to restart handler
                    msg.exec_()
                    return
            
            if len(face_locations) > 1:
                self.status_label.setText("Multiple faces detected. Please ensure only one face is visible.")
            else:
                self.status_label.setText("Position your face in the frame...")
        
        # Convert and display frame
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_feed.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_feed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def auto_capture_photo(self, frame):
        """Automatically capture photo when face is detected"""
        if self.camera and self.registration_camera_active:
            self.captured_photo = frame
            self.stop_registration_camera()
            self.status_label.setText("Great! Now enter the student's name below.")
            self.name_input.show()
            self.register_btn.show()
            
            # Show the captured photo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_feed.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.camera_feed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def register_student(self):
        """Register a new student with captured photo"""
        try:
            name = self.name_input.text().strip()
            if not name:
                raise ValueError("Please enter a student name")
            
            if not hasattr(self, 'captured_photo') or self.captured_photo is None:
                raise ValueError("Please capture a photo first")
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(self.captured_photo)
            if not face_encoding:
                raise ValueError("No face detected in photo. Please try again.")
            
            # Add student to database
            result = self.db_manager.add_student(
                name=name,
                face_embedding=face_encoding[0],
                course_id=self.current_course['_id'],
                photo_data=self.captured_photo
            )
            
            if result:
                # Show success message
                msg = self.create_styled_message_box(
                    QMessageBox.Information,
                    "Success",
                    "Student Registered",
                    f"{name} has been successfully registered."
                )
                msg.exec_()
                
                # Reset registration screen
                self.reset_registration_screen()
                
                # Reload face recognition data
                self.load_face_data()
                
                # Update student list
                self.update_student_list()
                self.refresh_analytics()  # Refresh analytics immediately
                
        except Exception as e:
            msg = self.create_styled_message_box(
                QMessageBox.Critical,
                "Registration Error",
                "Failed to register student",
                str(e)
            )
            # Connect QMessageBox's button to trigger a restart
            msg.buttonClicked.connect(self.handle_restart)
            msg.exec_()

    def load_face_data(self):
        """Load face embeddings for all students in the course"""
        face_data = self.db_manager.get_course_face_embeddings(self.current_course['_id'])
        for student_id, data in face_data.items():
            self.known_face_encodings[student_id] = data['embedding']
            self.known_face_names[student_id] = data['name'] 

    def create_styled_message_box(self, icon, title, text, informative_text="", buttons=None):
        """Create a theme-aware styled message box"""
        colors = self.base_styles['dark']
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.NoIcon)  # Remove icon
        msg.setWindowTitle(title)
        msg.setText(text)
        if informative_text:
            msg.setInformativeText(informative_text)
        
        if buttons:
            msg.setStandardButtons(buttons)
        
        # Improved styling with proper alignment and sizing
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['bg']};
                min-width: 300px;
                max-width: 300px;
            }}
            QMessageBox QLabel {{
                color: {colors['text']};
                font-size: 14px;
                padding: 10px;
                alignment: center;
            }}
            /* Main text */
            QMessageBox QLabel#qt_msgbox_label {{
                color: {colors['text']};
                font-size: 15px;
                font-weight: 500;
                padding-top: 20px;
            }}
            /* Informative text */
            QMessageBox QLabel#qt_msgbox_informativelabel {{
                color: {colors['secondary_text']};
                font-size: 13px;
                padding-bottom: 20px;
            }}
            /* Button styling */
            QMessageBox QPushButton {{
                background-color: #e0e0e0;
                color: #2c2c2e;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
                font-size: 13px;
                margin: 0px 5px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {colors['accent']};
                color: white;
            }}
        """)
        
        # Center align all labels
        for label in msg.findChildren(QLabel):
            label.setAlignment(Qt.AlignCenter)
        
        return msg

    def show_attendance_notification(self, name):
        """Show a temporary notification for attendance"""
        msg = self.create_styled_message_box(
            QMessageBox.Information,
            "Attendance Recorded",
            f"Attendance recorded for {name}"
        )
        QTimer.singleShot(2000, msg.accept)
        msg.exec_()

    def check_hand_raise(self, student_id, name, landmarks):
        """Check if a student is raising their hand"""
        # Get relevant landmarks
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Check if hand is raised (wrist above shoulder)
        if wrist.y < shoulder.y:
            current_time = time.time()
            cooldown_time = 5  # 5 seconds cooldown between hand raises
            
            # Check cooldown
            if (student_id not in self.last_hand_raise or 
                current_time - self.last_hand_raise.get(student_id, 0) > cooldown_time):
                
                self.last_hand_raise[student_id] = current_time
                self.log_hand_raise(student_id, name)
                
                # Start recording for question
                if not self.is_recording:
                    self.start_question_recording(student_id, name) 

    def start_question_recording(self, student_id, name):
        """Start recording audio for question analysis"""
        self.is_recording = True
        self.current_question_student = (student_id, name)
        
        # Show recording indicator on main thread
        QTimer.singleShot(0, lambda: self.show_recording_notification(name))
        
        # Start recording in a separate thread
        threading.Thread(target=self.record_audio, daemon=True).start()

    def show_recording_notification(self, name):
        """Show recording notification on main thread"""
        msg = self.create_styled_message_box(
            QMessageBox.Information,
            "Recording Question",
            f"Recording question from {name}",
            "Please speak clearly into the microphone."
        )
        QTimer.singleShot(5000, msg.accept)
        msg.show()

    def record_audio(self):
        """Record audio from microphone"""
        CHUNK = 2048  # Increased buffer size
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 10
        
        audio = None
        stream = None
        frames = []
        
        try:
            audio = pyaudio.PyAudio()
            
            # Get default input device info
            device_info = audio.get_default_input_device_info()
            logger.info(f"Using audio device: {device_info['name']}")
            
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_info['index'],
                stream_callback=None
            )
            
            logger.info("Started recording audio...")
            
            # Wait for stream to be ready
            time.sleep(0.1)
            
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not self.is_recording:
                    break
                    
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except OSError as e:
                    if e.errno == -9981:  # Input overflow
                        logger.warning("Audio buffer overflow, continuing recording...")
                        continue
                    else:
                        logger.error(f"Error recording audio: {str(e)}")
                        break
                except Exception as e:
                    logger.error(f"Unexpected error recording audio: {str(e)}")
                    break
                    
            logger.info("Finished recording audio")
            
        except Exception as e:
            logger.error(f"Error setting up audio recording: {str(e)}")
            
        finally:
            # Clean up resources
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {str(e)}")
                    
            if audio is not None:
                try:
                    audio.terminate()
                except Exception as e:
                    logger.error(f"Error terminating audio: {str(e)}")
            
            # Process recorded audio if we got any frames
            if frames:
                self.process_question_audio(frames)
            else:
                logger.error("No audio data recorded")
                QTimer.singleShot(0, lambda: self.show_error_message(
                    "Failed to record audio. Please try again."
                ))

    def process_question_audio(self, frames):
        """Process recorded audio"""
        try:
            logger.info("Processing recorded audio...")
            
            # Get current student info
            if not hasattr(self, 'current_question_student'):
                logger.error("No student selected for question")
                raise Exception("No student selected")
            
            student_id, name = self.current_question_student
            
            # Ensure we have a valid student ID
            if not student_id:
                logger.error("Invalid student ID")
                raise Exception("Invalid student ID")
            
            # Save frames to a temporary WAV file
            temp_wav = "temp_question.wav"
            wf = wave.open(temp_wav, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            logger.info("Sending audio to OpenAI for transcription...")
            
            # Open the file and send to OpenAI
            with open(temp_wav, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
            
            logger.info(f"Transcribed text: {transcript.text}")
            
            # Delete temporary file
            import os
            os.remove(temp_wav)
            
            logger.info("Analyzing question relevance...")
            
            # Get class topic from course description or name
            class_topic = (
                self.current_course.get('description', '') or 
                self.current_course['course_name']
            )
            
            # Analyze question relevance using GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing a student's question for relevance to a specific class topic. "
                            "You must determine whether the question is relevant to the class topic or not. "
                            "Answer either 'Relevant' or 'Irrelevant'. Then provide a brief explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The class topic is: {class_topic}\n"
                            f"Question: {transcript.text}\n\n"
                            "Is this question relevant or irrelevant to the class topic? "
                            "Respond strictly with one of the words 'Relevant' or 'Irrelevant' followed by a one-sentence justification."
                        )
                    }
                ]
            )
            
            response_text = response.choices[0].message.content
            # Split into relevance and reason
            relevance_word, *reason_parts = response_text.split(' ', 1)
            is_relevant = relevance_word.lower().startswith('relevant')
            reason = reason_parts[0] if reason_parts else ""
            
            # Log the question with the reason and student info
            self.log_question(student_id, name, transcript.text, is_relevant, reason)
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            QTimer.singleShot(0, lambda: self.show_error_message(str(e)))
        
        finally:
            self.is_recording = False

    def show_error_message(self, error_text):
        """Show error message on main thread"""
        msg = self.create_styled_message_box(
            QMessageBox.Warning,
            "Question Recording Failed",
            "Failed to process the question",
            error_text
        )
        msg.exec_()

    def log_hand_raise(self, student_id, name):
        """Log a hand raise event to the database"""
        try:
            self.db_manager.log_hand_raise(
                student_id=student_id,
                course_id=self.current_course['_id']
            )
            
            # Show notification
            msg = self.create_styled_message_box(
                QMessageBox.Information,
                "Hand Raise Detected",
                f"Hand raise recorded for {name}",
                "Recording question..."
            )
            QTimer.singleShot(2000, msg.accept)  # Auto-close after 2 seconds
            msg.show()
            
        except Exception as e:
            logger.error(f"Error logging hand raise for {name}: {str(e)}")

    def log_question(self, student_id, name, question_text, is_relevant, reason=""):
        """Log a question to the database"""
        try:
            logger.info(f"Attempting to log question for student {student_id} ({name})")
            
            result = self.db_manager.log_question(
                student_id=student_id,
                course_id=self.current_course['_id'],
                question_text=question_text,
                is_relevant=is_relevant,
                reason=reason
            )
            
            if result and result.inserted_id:
                logger.info(f"Question successfully logged with ID: {result.inserted_id}")
                
                def show_notification():
                    relevance_display = "Relevant" if is_relevant else "Not relevant"
                    msg = self.create_styled_message_box(
                        QMessageBox.Information,
                        "Question Recorded",
                        f"Question from {name} recorded",
                        f"Question: {question_text}\nRelevance: {relevance_display}"
                    )
                    QTimer.singleShot(3000, msg.accept)
                    msg.show()
                    
                    # Update UI components
                    self.update_engagement_list()
                    
                    # Force analytics tab refresh
                    analytics_tab = self.findChild(QWidget, 'analytics_tab')
                    if analytics_tab:
                        # Clear existing widgets
                        for i in reversed(range(analytics_tab.layout().count())): 
                            widget = analytics_tab.layout().itemAt(i).widget()
                            if widget:
                                widget.deleteLater()
                        # Reinitialize analytics tab
                        self._setup_analytics_tab(analytics_tab)
                
                QTimer.singleShot(0, show_notification)
                
            else:
                logger.error("Failed to get confirmation of question logging")
                raise Exception("Question logging failed")
            
        except Exception as e:
            logger.error(f"Error logging question for {name}: {str(e)}")
            # Show error on main thread
            QTimer.singleShot(0, lambda: self.show_error_message(str(e)))

    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.dark_mode = not self.dark_mode
        self.update_dark_mode_button()
        self.update_theme()

    def update_theme(self):
        """Update the application theme"""
        colors = self.base_styles['dark']
        self.setStyleSheet(self.get_style())
        
        # Update registration buttons if they exist
        if hasattr(self, 'start_camera_btn'):
            self.start_camera_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {colors['accent']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {colors['accent_hover']};
                }}
                QPushButton:disabled {{
                    background-color: {colors['border']};
                    color: {colors['secondary_text']};
                }}
            """)
            
            self.capture_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {colors['success']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: #2fb344;
                }}
                QPushButton:disabled {{
                    background-color: {colors['border']};
                    color: {colors['secondary_text']};
                }}
            """)
            
            self.register_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {colors['accent']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {colors['accent_hover']};
                }}
                QPushButton:disabled {{
                    background-color: {colors['border']};
                    color: {colors['secondary_text']};
                }}
            """)
            
            self.name_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {colors['bg']};
                    color: {colors['text']};
                    border: 2px solid {colors['border']};
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                    margin-top: 20px;
                }}
                QLineEdit:focus {{
                    border-color: {colors['accent']};
                }}
                QLineEdit::placeholder {{
                    color: {colors['secondary_text']};
                }}
            """)
        
        # Update other widgets
        self.update_engagement_list()
        self.update_student_list()

    def update_dark_mode_button(self):
        """Update dark mode button icon"""
        colors = self.base_styles['dark']
        
        self.dark_mode_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 20px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
            }}
        """)
        
        # Set icon based on current mode
        icon_color = "white" if self.dark_mode else "#1d1d1f"
        self.dark_mode_button.setText("🌙" if self.dark_mode else "☀️")

    def create_question_card(self, question):
        """Create a clean, minimalist question card"""
        card = QWidget()
        colors = self.base_styles['dark']
        
        # Set the background color to #2c2c2e
        card.setStyleSheet(f"""
            QWidget {{
                background-color: transparent; /* Set to #2c2c2e */
                border-radius: 8px;  /* Optional: Add rounded corners */
                padding: 16px;
                margin-bottom: 10px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Question text
        text = QLabel(question['question_text'])
        text.setWordWrap(True)
        text.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 500;
            color: {colors['text']};
        """)
        
        # Metadata row
        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        
        # Format timestamp
        timestamp = question['timestamp']
        time_str = f"Asked by {question.get('student_name', 'Student')} on {timestamp.strftime('%d/%B')} at {timestamp.strftime('%H:%M')}"
        time = QLabel(time_str)
        time.setStyleSheet(f"""
            color: {colors['secondary_text']};
            font-size: 13px;
        """)
        
        # Relevance badge
        is_relevant = question['is_relevant']
        relevance = QLabel("Relevant" if is_relevant else "Irrelevant")
        relevance.setStyleSheet(f"""
            background-color: {colors['success'] if is_relevant else colors['error']};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 500;
        """)
        
        meta.addWidget(time)
        meta.addStretch()
        meta.addWidget(relevance)
        
        layout.addWidget(text)
        layout.addLayout(meta)
        
        # Add reason if available
        if 'reason' in question:
            reason = QLabel(f"Reason: {question['reason']}")
            reason.setWordWrap(True)
            reason.setStyleSheet(f"""
                color: {colors['secondary_text']};
                font-size: 13px;
                font-style: italic;
                padding-top: 4px;
            """)
            layout.addWidget(reason)
        
        return card 

    def stop_registration_camera(self):
        """Stop the registration camera preview"""
        if hasattr(self, 'camera') and self.camera:
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None
        
        self.registration_camera_active = False
        
        # Clear the camera feed if it exists and no photo was captured
        if hasattr(self, 'camera_feed') and not hasattr(self, 'captured_photo'):
            self.camera_feed.clear()
            # Remove this line that sets the text back to "Camera Preview"
            # self.camera_feed.setText("Camera Preview")

    def reset_registration_screen(self):
        """Reset registration screen to initial state"""
        self.registration_container.hide()
        self.welcome_container.show()
        self.name_input.clear()
        self.name_input.hide()
        self.register_btn.hide()
        self.captured_photo = None
        self.camera_feed.clear()
        # Remove this line that sets the text back to "Camera Preview"
        # self.camera_feed.setText("Camera Preview")
        self.status_label.setText("") 

    def refresh_analytics(self):
        """Refresh the analytics tab"""
        analytics_tab = self.findChild(QWidget, 'analytics_tab')
        if analytics_tab and analytics_tab.isVisible():
            # Get the existing layout
            existing_layout = analytics_tab.layout()
            if existing_layout:
            # Clear existing widgets
                while existing_layout.count():
                    item = existing_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Repopulate the existing layout
                self._populate_analytics_tab(existing_layout)
            else:
                # Create new layout if none exists
                self._setup_analytics_tab(analytics_tab) 

    # Add this new method
    def toggle_camera(self):
        """Toggle camera for registration"""
        if not hasattr(self, 'camera') or self.camera is None:
            # Reset variables when starting camera
            self.reset_registration_variables()
            
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.registration_camera_active = True
                self.camera_timer.start(30)
                self.start_camera_btn.setText("Stop Camera")
            else:
                msg = self.create_styled_message_box(
                    QMessageBox.Critical,
                    "Error",
                    "Could not access the camera."
                )
                msg.exec_()
        else:
            self.stop_registration_camera()
            self.start_camera_btn.setText("Start Camera")

    def on_tab_changed(self, index):
        """Handle tab changes and cleanup resources"""
        old_widget = self.tab_widget.widget(self.previous_tab_index)
        new_widget = self.tab_widget.widget(index)

        # If the old tab was attendance_tab, do any necessary cleanup:
        if old_widget and old_widget.objectName() == "attendance_tab":
            # Optionally clean up camera or other resources here
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
                self.camera = None
                if hasattr(self, 'camera_timer'):
                    self.camera_timer.stop()

            # Clear any attendance feed if you like,
            # but do not call reset_attendance_screen() here
            # unless you specifically want to reset on departure too.

        # If old tab was registration_tab, reset it:
        if old_widget and old_widget.objectName() == "registration_tab":
            self.reset_registration_screen()

        # We also generally clear the camera feed any time we change tabs:
        if hasattr(self, 'camera_feed'):
            self.camera_feed.clear()
        if hasattr(self, 'camera_label'):
            self.camera_label.clear()

        # Identify new tab
        current_tab = new_widget.objectName() if new_widget else ""

        # When arriving at the attendance tab, ALWAYS reset it to default:
        if current_tab == "attendance_tab":
            self.reset_attendance_screen()

        # Refresh engagement or analytics only if relevant
        if current_tab == "engagement_tab":
            logger.info("Refreshing engagement tab...")
            QTimer.singleShot(0, self.update_engagement_list)
        elif current_tab == "analytics_tab":
            logger.info("Refreshing analytics tab...")
            QTimer.singleShot(0, lambda: self.refresh_analytics())

        # Update stored tab index
        self.previous_tab_index = index

    def handle_registration_success(self):
        """Handle successful student registration"""
        # Existing registration success code...
        
        # Ensure camera is properly cleaned up
        if self.camera:
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None
        
        # Reset camera preview
        self.camera_feed.clear()
        # Remove this line that sets the text back to "Camera Preview"
        # self.camera_feed.setText("Camera Preview")
        
        # Reset registration UI state
        self.registration_camera_active = False
        self.face_detection_attempts = None
        self.captured_photo = None
        
        # Show success message
        msg = self.create_styled_message_box(
            QMessageBox.Information,
            "Registration Complete",
            "Student successfully registered!"
        )
        msg.exec_()

    def handle_restart(self):
        """Handle restart after error by resetting to initial registration state"""
        logger.info("Starting complete registration reset process...")
        
        # Stop and cleanup camera
        if hasattr(self, 'camera') and self.camera:
            logger.info("Stopping camera...")
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None
        
        # Reset all registration variables
        self.face_detection_attempts = 0
        self.registration_camera_active = False
        self.captured_photo = None
        
        # Clear UI elements
        self.camera_feed.clear()
        self.status_label.setText("")
        
        # Hide registration container and show welcome screen
        logger.info("Resetting to welcome screen...")
        self.registration_container.hide()
        self.welcome_container.show()
        
        # Hide input fields if they're visible
        if hasattr(self, 'name_input'):
            self.name_input.hide()
            self.name_input.clear()
        
        if hasattr(self, 'register_btn'):
            self.register_btn.hide()
        
        logger.info("Registration reset complete")

    def _get_top_students(self, questions, relevant=True):
        """Get top 3 students based on question relevance"""
        from collections import defaultdict
        
        # Count questions per student
        student_counts = defaultdict(int)
        for q in questions:
            if q['is_relevant'] == relevant:
                student_counts[q['student_name']] += 1
        
        # Sort and get top 3
        top_students = sorted(
            student_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return top_students

    def create_pie_chart(self, relevant_count, irrelevant_count):
        """Create a pie chart with proper scaling and proportions"""
        # Validate input data
        if relevant_count < 0 or irrelevant_count < 0:
            relevant_count = 0
            irrelevant_count = 0
        
        # Ensure at least one value is non-zero to prevent division by zero
        if relevant_count == 0 and irrelevant_count == 0:
            relevant_count = 1  # Default to show 100% relevant if no data
        
        # Create figure with fixed aspect ratio and size
        fig = Figure(figsize=(8, 8))  # Make it square with larger size
        fig.patch.set_facecolor('#323232')  # Match app background
        
        # Add subplot with equal aspect ratio
        ax = fig.add_subplot(111, aspect='equal')  # Force equal aspect ratio
        ax.set_facecolor('#323232')
        
        # Data
        sizes = [relevant_count, irrelevant_count]
        colors = ['#30d158', '#ff453a']  # Success and error colors
        
        try:
            # Create pie chart with specific layout parameters
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=None,
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                startangle=90,
                wedgeprops=dict(edgecolor='none'),
            )
            
            # Add legend with white text
            legend = ax.legend(
                wedges,
                ['Relevant', 'Irrelevant'],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )
            plt.setp(legend.get_texts(), color='white', fontsize=12)
            
            # Style percentage text
            plt.setp(autotexts, color='white', size=12, weight='bold')
            
            # Turn off axis
            ax.axis('off')
            
            # Create canvas with fixed size
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(400, 400)  # Set fixed size for the widget
            
            # Adjust layout to prevent text cutoff
            fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust rect to accommodate legend
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            # Create a simple text message if chart creation fails
            ax.text(0.5, 0.5, 'No data available',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    fontsize=14)
            ax.axis('off')
        
        return canvas

    def reset_registration_variables(self):
        """Reset all registration-related variables"""
        self.face_detection_attempts = 0
        self.registration_camera_active = False
        self.captured_photo = None
        
        # Clear UI elements if they exist
        if hasattr(self, 'camera_feed'):
            self.camera_feed.clear()
        if hasattr(self, 'status_label'):
            self.status_label.setText("")

    def show_course_details(self):
        """Show course details dialog"""
        dialog = CourseDetailsDialog(self, self.db_manager, self.current_course, self.base_styles['dark'])
        dialog.exec_()

    def edit_course(self):
        """Handle course name editing in CourseDetailsDialog"""
        logger.info("Edit button clicked in CourseDetailsDialog")
        
        # Create edit dialog
        edit_dialog = QDialog(self)
        edit_dialog.setWindowTitle("Edit Course")
        edit_dialog.setFixedWidth(300)
        edit_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.colors['bg']};
                border-radius: 10px;
                padding: 20px;
            }}
            QLabel {{
                color: {self.colors['text']};
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }}
            QPushButton {{
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton#save_btn {{
                background-color: {self.colors['accent']};
                color: white;
                border: none;
            }}
            QPushButton#save_btn:hover {{
                background-color: {self.colors['accent_hover']};
            }}
            QPushButton#cancel_btn {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
            }}
            QPushButton#cancel_btn:hover {{
                background-color: {self.colors['border']};
            }}
        """)
        
        layout = QVBoxLayout(edit_dialog)
        
        # Course name input
        name_label = QLabel("Course Name:")
        name_input = QLineEdit(self.course['course_name'])
        logger.info(f"Current course name: {self.course['course_name']}")
        
        # Course code input
        code_label = QLabel("Course Code:")
        code_input = QLineEdit(self.course['course_code'])
        logger.info(f"Current course code: {self.course['course_code']}")
        
        # Add inputs to layout
        layout.addWidget(name_label)
        layout.addWidget(name_input)
        layout.addWidget(code_label)
        layout.addWidget(code_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setObjectName("save_btn")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel_btn")
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        def save_changes():
            logger.info("Save button clicked")
            new_name = name_input.text().strip()
            new_code = code_input.text().strip()
            logger.info(f"New name: {new_name}, New code: {new_code}")
            
            if new_name and new_code:
                try:
                    logger.info("Attempting to update course in database")
                    # Update in database with correct parameter names
                    self.db_manager.update_course(
                        course_id=self.course['_id'],
                        course_name=new_name,  # Changed from new_name
                        course_code=new_code   # Changed from new_code
                    )
                    
                    # Update local course data
                    self.course['course_name'] = new_name
                    self.course['course_code'] = new_code
                    logger.info("Course data updated locally")
                    
                    # Update parent window's course display
                    self.parent().update_course_display()
                    logger.info("Parent window display updated")
                    
                    # Update dialog title
                    for widget in self.findChildren(QLabel):
                        if widget.text().startswith(f"{self.course['course_name']} {self.course['course_code']}"):
                            widget.setText(f"{new_name} {new_code}")
                            break
                    logger.info("Dialog title updated")
                    
                    # Show success message
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Success")
                    msg.setText("Course updated successfully")
                    msg.setStyleSheet(edit_dialog.styleSheet())
                    msg.exec_()
                    
                    edit_dialog.accept()
                    self.accept()  # Close both dialogs
                    
                except Exception as e:
                    logger.error(f"Error updating course: {str(e)}")
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Error")
                    msg.setText(f"Failed to update course: {str(e)}")
                    msg.setStyleSheet(edit_dialog.styleSheet())
                    msg.exec_()
            else:
                logger.warning("Invalid input: Empty name or code")
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Invalid Input")
                msg.setText("Please enter both course name and code")
                msg.setStyleSheet(edit_dialog.styleSheet())
                msg.exec_()
        
        save_btn.clicked.connect(save_changes)
        cancel_btn.clicked.connect(edit_dialog.reject)
        
        logger.info("Showing edit dialog")
        edit_dialog.exec_()

    def update_course_display(self):
        """Update the course display after editing"""
        course_name = self.current_course['course_name'].strip().upper()
        course_id = self.current_course['course_code'].strip()
        new_course_display = f"{course_name}{course_id}"
        
        # Find and update the course info label
        for widget in self.findChildren(QLabel):
            if hasattr(self, 'current_course_display') and widget.text() == self.current_course_display:
                widget.setText(new_course_display)
                self.current_course_display = new_course_display
                logger.info(f"Updated course display to: {new_course_display}")
                break
            # Fallback if current_course_display doesn't match
            elif widget.text().startswith(course_name) or widget.text().endswith(course_id):
                widget.setText(new_course_display)
                self.current_course_display = new_course_display
                logger.info(f"Updated course display using fallback to: {new_course_display}")
                break

    def update_attendance_display(self):
        today = datetime.now().date()
        attendance_records = self.db_manager.get_attendance_records(
            course_id=self.current_course['_id'],
            date_param=today  # Changed from 'date' to 'date_param'
        )
        # ... rest of the method ...

    def create_hand_raises_chart(self, students):
        """Create a horizontal bar chart showing hand raises per student"""
        # Create figure with larger width for labels
        fig = Figure(figsize=(10, 8))
        fig.patch.set_facecolor('#323232')
        
        # Add subplot
        ax = fig.add_subplot(111)
        ax.set_facecolor('#323232')
        
        # Get hand raises data from database and sort it
        student_data = []
        for student in students:
            hand_raises = self.db_manager.get_student_hand_raises(
                student['student_id'],
                self.current_course['_id']
            )
            if hand_raises > 0:
                student_data.append((student['name'], hand_raises))
        
        # Sort by number of hand raises (descending)
        student_data.sort(key=lambda x: x[1], reverse=True)
        names = [data[0] for data in student_data]
        hand_raises = [data[1] for data in student_data]
        
        # Create canvas with fixed size
        canvas = FigureCanvas(fig)
        canvas.setFixedSize(500, 400)
        
        try:
            if student_data:
                # Create horizontal bars with reduced height and spacing
                y_pos = np.arange(len(names)) * 0.3  # Reduced from 0.6 to 0.3 to halve the spacing
                bars = ax.barh(y_pos, hand_raises, 
                             color='#30d158',
                             height=0.2)
                
                # Remove axes and grid
                ax.set_frame_on(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(True)
                ax.grid(False)
                
                # Set student names as y-axis labels with more space
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, color='white', fontsize=10)
                
                # Add value labels at the end of each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}',
                           ha='left', va='center',
                           color='white',
                           fontsize=12,
                           fontweight='bold')
            
            else:
                ax.text(0.5, 0.5, 'No hand raises recorded',
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white',
                        fontsize=14,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Adjust margins to prevent label cutoff
            fig.subplots_adjust(left=0.3, right=0.9)
            
        except Exception as e:
            logger.error(f"Error creating hand raises chart: {e}")
            ax.text(0.5, 0.5, 'Error creating chart',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    fontsize=14)
            ax.axis('off')
        
        return canvas

    def exit_to_menu(self):
        """
        Closes this window and shows the course selection again,
        allowing the user to choose a new course.
        """
        self.close()

        course_dialog = CourseSelectionDialog(self.db_manager, None)

        # Again call the helper function to center the dialog
        def center_dialog_on_screen(dialog):
            dialog.adjustSize()
            qr = dialog.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            dialog.move(qr.topLeft())

        center_dialog_on_screen(course_dialog)

        if course_dialog.exec_() == QDialog.Accepted:
            new_course = course_dialog.get_selected_course()
            if not new_course:
                # User canceled or no valid course selected
                return
            # Pass that new course to MainWindow so it won't prompt again
            new_window = MainWindow(self.db_manager, self.config, selected_course=new_course)
            # Show the new main window maximized to fill the screen
            new_window.showMaximized()

    # ... existing code...
    ### Insert this new helper method to restore attendance tab state
    def reset_attendance_screen(self):
        """
        Revert the Attendance tab back to its default "Start Class Recording" state.
        """
        # Stop camera if running
        if hasattr(self, 'camera') and self.camera:
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None

        # Reset any tracking variables
        self.is_class_recording = False
        if hasattr(self, 'attendance_recorded'):
            self.attendance_recorded.clear()
        
        # Hide the active attendance container, show the welcome container
        if hasattr(self, 'attendance_container'):
            self.attendance_container.hide()            
        if hasattr(self, 'attendance_welcome_container'):
            self.attendance_welcome_container.show()
        
        # Reset the status label, and re-enable/disable buttons
        if hasattr(self, 'attendance_status'):
            self.attendance_status.setText("")
        if hasattr(self, 'start_recording_button'):
            self.start_recording_button.setEnabled(True)
            self.start_recording_button.setText("Start Class Recording")
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(False)
        
        # Clear the camera feed if it exists
        if hasattr(self, 'camera_label'):
            self.camera_label.clear()

class CircularTimer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.value = 0  # 0 to 360
        self.number = 3
        
        # Get colors
        colors = parent.base_styles['dark']
        self.colors = colors
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate circle parameters
        width = self.width()
        height = self.height()
        rect = QRectF(10, 10, width-20, height-20)
        
        # Draw background circle (gray outline)
        pen = QPen(QColor(self.colors['border']))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawEllipse(rect)
        
        # Draw progress arc (blue) - now 3x thicker
        pen.setColor(QColor(self.colors['accent']))
        pen.setWidth(9)  # Changed from 3 to 9
        painter.setPen(pen)
        painter.drawArc(rect, 90*16, -self.value*16)  # Start from top (90°), go clockwise
        
        # Draw number
        painter.setPen(QColor(self.colors['accent']))
        font = painter.font()
        font.setPointSize(50)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(self.number))

class CourseDetailsDialog(QDialog):
    def __init__(self, parent, db_manager, course, colors):
        super().__init__(parent)
        self.db_manager = db_manager
        self.course = course  # This is the course data passed in
        self.colors = colors
        self.parent_window = parent
        
        self.setWindowTitle("Course Details")
        self.setFixedWidth(400)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['bg']};
                border-radius: 10px;
                padding: 20px;
            }}
            QLabel {{
                color: {colors['text']};
                font-size: 14px;
            }}
            QPushButton {{
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton#edit_btn {{
                background-color: {colors['bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
            }}
            QPushButton#edit_btn:hover {{
                background-color: {colors['border']};
            }}
            QPushButton#done_btn {{
                background-color: {colors['accent']};
                color: white;
                border: none;
            }}
            QPushButton#done_btn:hover {{
                background-color: {colors['accent_hover']};
            }}
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Course Title
        title = QLabel(f"{self.course['course_name']} {self.course['course_code']}")
        title.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {self.colors['text']};
            padding-bottom: 10px;
        """)
        layout.addWidget(title)
        
        # Get course statistics
        students = self.db_manager.get_course_students(self.course['_id'])  # Use self.course instead of self.current_course
        logger.info(f"Retrieved {len(students)} students for course {self.course['_id']}")  # Fixed here too
        questions = self.db_manager.get_course_questions(self.course['_id'])
        
        # Calculate engagement ratio
        relevant_count = sum(1 for q in questions if q.get('is_relevant', True))
        irrelevant_count = len(questions) - relevant_count
        ratio = f"{relevant_count}:{irrelevant_count}" if len(questions) > 0 else "No questions yet"
        
        # Student Count
        student_count = QLabel(f"Number of Students: {len(students)}")
        student_count.setStyleSheet("padding: 5px 0;")
        layout.addWidget(student_count)
        
        # Engagement Ratio
        engagement = QLabel(f"Engagement Ratio: {ratio}")
        engagement.setStyleSheet("padding: 5px 0;")
        layout.addWidget(engagement)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        edit_btn = QPushButton("Edit")
        edit_btn.setObjectName("edit_btn")
        edit_btn.clicked.connect(self.edit_course)
        
        done_btn = QPushButton("Done")
        done_btn.setObjectName("done_btn")
        done_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(edit_btn)
        button_layout.addWidget(done_btn)
        
        layout.addStretch()
        layout.addLayout(button_layout)
    
    def edit_course(self):
        """Handle course name editing"""
        logger.info("Edit button clicked in CourseDetailsDialog")
        
        # Create edit dialog
        edit_dialog = QDialog(self)
        edit_dialog.setWindowTitle("Edit Course")
        edit_dialog.setFixedWidth(300)
        edit_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.colors['bg']};
                border-radius: 10px;
                padding: 20px;
            }}
            QLabel {{
                color: {self.colors['text']};
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }}
            QPushButton {{
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton#save_btn {{
                background-color: {self.colors['accent']};
                color: white;
                border: none;
            }}
            QPushButton#save_btn:hover {{
                background-color: {self.colors['accent_hover']};
            }}
            QPushButton#cancel_btn {{
                background-color: {self.colors['bg']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
            }}
            QPushButton#cancel_btn:hover {{
                background-color: {self.colors['border']};
            }}
        """)
        
        layout = QVBoxLayout(edit_dialog)
        
        # Course name input
        name_label = QLabel("Course Name:")
        name_input = QLineEdit(self.course['course_name'])
        logger.info(f"Current course name: {self.course['course_name']}")
        
        # Course code input
        code_label = QLabel("Course Code:")
        code_input = QLineEdit(self.course['course_code'])
        logger.info(f"Current course code: {self.course['course_code']}")
        
        # Add inputs to layout
        layout.addWidget(name_label)
        layout.addWidget(name_input)
        layout.addWidget(code_label)
        layout.addWidget(code_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setObjectName("save_btn")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel_btn")
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        def save_changes():
            logger.info("Save button clicked")
            new_name = name_input.text().strip()
            new_code = code_input.text().strip()
            logger.info(f"New name: {new_name}, New code: {new_code}")
            
            if new_name and new_code:
                try:
                    logger.info("Attempting to update course in database")
                    # Update in database with correct parameter names
                    self.db_manager.update_course(
                        course_id=self.course['_id'],
                        course_name=new_name,  # Changed from new_name
                        course_code=new_code   # Changed from new_code
                    )
                    
                    # Update local course data
                    self.course['course_name'] = new_name
                    self.course['course_code'] = new_code
                    logger.info("Course data updated locally")
                    
                    # Update parent window's course display
                    self.parent_window.update_course_display()
                    logger.info("Parent window display updated")
                    
                    # Update dialog title
                    for widget in self.findChildren(QLabel):
                        if widget.text().startswith(f"{self.course['course_name']} {self.course['course_code']}"):
                            widget.setText(f"{new_name} {new_code}")
                            break
                    logger.info("Dialog title updated")
                    
                    # Show success message
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Success")
                    msg.setText("Course updated successfully")
                    msg.setStyleSheet(edit_dialog.styleSheet())
                    msg.exec_()
                    
                    edit_dialog.accept()
                    self.accept()  # Close both dialogs
                    
                except Exception as e:
                    logger.error(f"Error updating course: {str(e)}")
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Error")
                    msg.setText(f"Failed to update course: {str(e)}")
                    msg.setStyleSheet(edit_dialog.styleSheet())
                    msg.exec_()
            else:
                logger.warning("Invalid input: Empty name or code")
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Invalid Input")
                msg.setText("Please enter both course name and code")
                msg.setStyleSheet(edit_dialog.styleSheet())
                msg.exec_()
        
        save_btn.clicked.connect(save_changes)
        cancel_btn.clicked.connect(edit_dialog.reject)
        
        logger.info("Showing edit dialog")
        edit_dialog.exec_()

    def create_hand_raises_chart(self, students):
        """Create a bar chart showing hand raises per student"""
        # Create figure with matching background
        fig = Figure(figsize=(8, 8))
        fig.patch.set_facecolor('#323232')
        
        # Add subplot
        ax = fig.add_subplot(111)
        ax.set_facecolor('#323232')
        
        # Get hand raises data from database
        student_hand_raises = []
        names = []
        
        for student in students:
            hand_raises = self.db_manager.get_student_hand_raises(
                student['student_id'],
                self.current_course['_id']
            )
            if hand_raises > 0:  # Only show students who raised hands
                student_hand_raises.append(hand_raises)
                names.append(student['name'])
        
        try:
            if student_hand_raises:
                # Create bars
                bars = ax.bar(names, student_hand_raises, color='#30d158')
                
                # Customize appearance
                ax.set_ylabel('Number of Hand Raises', color='white', fontsize=10)
                ax.set_title('Hand Raises per Student', color='white', fontsize=12, pad=20)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Style the chart
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.tick_params(colors='white')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', color='white')
                
            else:
                # Show message if no hand raises
                ax.text(0.5, 0.5, 'No hand raises recorded',
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white',
                        fontsize=14,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Remove grid
            ax.grid(False)
            
            # Create canvas with fixed size
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(400, 400)
            
            # Adjust layout to prevent text cutoff
            fig.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating hand raises chart: {e}")
            ax.text(0.5, 0.5, 'Error creating chart',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    fontsize=14)
            ax.axis('off')
        
        return canvas

def _populate_analytics_tab(self, layout):
    # ... existing code until charts section ...

    # Create a container for charts
    charts_container = QWidget()
    charts_layout = QHBoxLayout(charts_container)
    charts_layout.setSpacing(40)
    
    # Left side: Pie Chart
    pie_chart_container = QWidget()
    pie_layout = QVBoxLayout(pie_chart_container)
    pie_layout.setAlignment(Qt.AlignCenter)
    
    pie_chart_canvas = self.create_pie_chart(relevant_count, irrelevant_count)
    pie_layout.addWidget(pie_chart_canvas, alignment=Qt.AlignCenter)
    charts_layout.addWidget(pie_chart_container)
    
    # Right side: Hand Raises Bar Graph
    bar_chart_container = QWidget()
    bar_layout = QVBoxLayout(bar_chart_container)
    bar_layout.setAlignment(Qt.AlignCenter)
    
    bar_chart_canvas = self.create_hand_raises_chart(students)
    bar_layout.addWidget(bar_chart_canvas, alignment=Qt.AlignCenter)
    charts_layout.addWidget(bar_chart_container)
    
    # Add the charts container to main layout
    layout.addWidget(charts_container, alignment=Qt.AlignCenter)