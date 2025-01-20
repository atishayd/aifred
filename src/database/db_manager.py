from pymongo import MongoClient
import os
from dotenv import load_dotenv
from src.utils.logger import logger
import certifi
from datetime import datetime, date, timedelta
import cv2
import numpy as np

class DatabaseManager:
    def __init__(self):
        load_dotenv()
        
        # Construct MongoDB Atlas URI
        username = os.getenv('MONGODB_USERNAME')
        password = os.getenv('MONGODB_PASSWORD')
        cluster = os.getenv('MONGODB_CLUSTER')
        
        if not all([username, password, cluster]):
            raise ValueError("MongoDB Atlas credentials not found in environment variables")
        
        uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=aifred"
        
        try:
            self.client = MongoClient(uri, tlsCAFile=certifi.where())
            self.db = self.client['aifred']
            
            # Test connection
            self.db.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Verify collections
            self.verify_collections()
            
            self._setup_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def verify_collections(self):
        """Verify all required collections exist"""
        try:
            required_collections = ['students', 'courses', 'attendance', 'questions', 'hand_raises']
            existing = self.db.list_collection_names()
            
            for collection in required_collections:
                if collection not in existing:
                    logger.info(f"Creating {collection} collection...")
                    self.db.create_collection(collection)
                    
                    # Create indexes for the new collection
                    if collection == 'questions':
                        self.db.questions.create_index([
                            ('student_id', 1),
                            ('course_id', 1),
                            ('timestamp', -1)
                        ])
                        
            logger.info("All required collections verified")
            
        except Exception as e:
            logger.error(f"Error verifying collections: {str(e)}")
            raise

    def initialize(self):
        """Initialize the database collections and indexes"""
        try:
            # Create courses collection
            if 'courses' not in self.db.list_collection_names():
                self.db.create_collection('courses')
                self.db.courses.create_index('course_code', unique=True)
                logger.info("Created courses collection")
            
            # Update students collection to include course reference
            if 'students' not in self.db.list_collection_names():
                self.db.create_collection('students')
                self.db.students.create_index([
                    ('student_id', 1),
                    ('course_id', 1)
                ], unique=True)
                logger.info("Created students collection")
            
            if 'attendance' not in self.db.list_collection_names():
                self.db.create_collection('attendance')
                self.db.attendance.create_index([('date', 1), ('student_id', 1)])
                logger.info("Created attendance collection")
            
            if 'engagement' not in self.db.list_collection_names():
                self.db.create_collection('engagement')
                self.db.engagement.create_index('student_id')
                logger.info("Created engagement collection")
                
            # Add questions collection with proper logging
            if 'questions' not in self.db.list_collection_names():
                logger.info("Creating questions collection...")
                self.db.create_collection('questions')
                self.db.questions.create_index([
                    ('student_id', 1),
                    ('course_id', 1),
                    ('timestamp', -1)
                ])
                logger.info("Successfully created questions collection")
                
                # Verify collection creation
                if 'questions' in self.db.list_collection_names():
                    logger.info("Verified questions collection exists")
                else:
                    logger.error("Failed to create questions collection")
                
            # Add hand raises collection
            if 'hand_raises' not in self.db.list_collection_names():
                self.db.create_collection('hand_raises')
                self.db.hand_raises.create_index([
                    ('student_id', 1),
                    ('course_id', 1),
                    ('timestamp', -1)
                ])
                logger.info("Created hand raises collection")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
        
    def create_course(self, course_name, course_code, description=""):
        """Create a new course"""
        course = {
            'course_name': course_name,
            'course_code': course_code,
            'description': description,
            'created_at': datetime.utcnow()
        }
        return self.db.courses.insert_one(course)
    
    def get_all_courses(self):
        """Get all courses"""
        return list(self.db.courses.find())
    
    def get_course(self, course_id):
        """Get course by ID"""
        return self.db.courses.find_one({'_id': course_id})
    
    def add_student(self, name, face_embedding, course_id, photo_data):
        """Add a new student to a specific course"""
        try:
            # Get all students in this specific course
            course_students = list(self.db.students.find({'course_id': course_id}))
            
            # If there are no students in this course, start with ID 1
            if not course_students:
                student_id = 1
            else:
                # Find the highest student_id in this course
                max_id = max(student.get('student_id', 0) for student in course_students)
                student_id = max_id + 1
            
            # Encode photo data with quality parameter
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, buffer = cv2.imencode('.jpg', photo_data, encode_params)
            if buffer is None or len(buffer) == 0:
                raise ValueError("Failed to encode photo")
            
            photo_bytes = buffer.tobytes()
            
            # Create the student document
            student = {
                'student_id': student_id,
                'course_id': course_id,
                'name': name,
                'face_embedding': face_embedding.tobytes(),
                'photo': photo_bytes,
                'created_at': datetime.utcnow()
            }
            
            # Create a compound index for student_id and course_id if it doesn't exist
            try:
                self.db.students.create_index(
                    [('student_id', 1), ('course_id', 1)],
                    unique=True,
                    name='student_course_unique'
                )
            except Exception as e:
                logger.debug(f"Index already exists: {str(e)}")
            
            # Insert the student document
            result = self.db.students.insert_one(student)
            logger.info(f"Successfully added student {name} with ID {student_id} to course {course_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding student: {str(e)}")
            raise
    
    def get_course_students(self, course_id):
        """Get all students in a course"""
        try:
            return list(self.db.students.find({'course_id': course_id}))
        except Exception as e:
            logger.error(f"Error getting course students: {str(e)}")
            return []
        
    def get_student(self, student_id):
        """Retrieve a student by ID"""
        return self.db.students.find_one({'_id': student_id})
        
    def mark_attendance(self, student_id, date_param, status='Present', course_id=None):
        """Mark attendance for a student"""
        try:
            # Convert date to datetime at start of day if it's a date object
            if isinstance(date_param, date):
                date_start = datetime.combine(date_param, datetime.min.time())
            else:
                date_start = date_param.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if attendance already exists for today
            existing = self.db.attendance.find_one({
                'student_id': student_id,
                'course_id': course_id,
                'date': date_start
            })
            
            if existing:
                return existing
            
            # Get student name
            student = self.get_student_by_id(student_id)
            if not student:
                raise Exception("Student not found")
            
            # Create attendance record
            attendance = {
                'student_id': student_id,
                'course_id': course_id,
                'date': date_start,  # Using datetime object
                'status': status,
                'timestamp': datetime.utcnow(),  # Changed from datetime.datetime.utcnow()
                'student_name': student.get('name', 'Unknown')
            }
            return self.db.attendance.insert_one(attendance)
            
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            raise
        
    def log_engagement(self, student_id, hand_raises=0, relevant_questions=0):
        """Log student engagement metrics"""
        engagement = {
            'student_id': student_id,
            'hand_raises': hand_raises,
            'relevant_questions': relevant_questions,
            'timestamp': datetime.utcnow()
        }
        return self.db.engagement.insert_one(engagement)
        
    def get_course_by_code(self, course_code):
        """Get course by course code"""
        return self.db.courses.find_one({'course_code': course_code})
        
    def update_course(self, course_id, course_name=None, course_code=None, description=None):
        """Update course details"""
        update_data = {}
        if course_name:
            update_data['course_name'] = course_name
        if course_code:
            update_data['course_code'] = course_code
        if description:
            update_data['description'] = description
        
        if update_data:
            return self.db.courses.update_one(
                {'_id': course_id},
                {'$set': update_data}
            )
        
    def delete_course(self, course_id):
        """Delete course and its students"""
        # Start a session for atomic operations
        with self.client.start_session() as session:
            with session.start_transaction():
                # Delete all students in the course
                self.db.students.delete_many({'course_id': course_id})
                # Delete the course
                result = self.db.courses.delete_one({'_id': course_id})
                return result 
        
    def get_course_face_embeddings(self, course_id):
        """Get face embeddings for all students in a course"""
        students = self.get_course_students(course_id)
        face_data = {}
        for student in students:
            # Convert bytes back to numpy array
            embedding = np.frombuffer(student['face_embedding'], dtype=np.float64)
            face_data[student['student_id']] = {
                'name': student['name'],
                'embedding': embedding
            }
        return face_data
        
    def get_today_attendance(self, course_id):
        """Get attendance records for today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return list(self.db.attendance.find({
            'course_id': course_id,
            'date': today
        })) 
        
    def remove_student(self, student_id, course_id):
        """Remove a student and all their associated data from a course"""
        with self.client.start_session() as session:
            with session.start_transaction():
                try:
                    # Remove student record
                    result = self.db.students.delete_one({
                        'student_id': student_id,
                        'course_id': course_id
                    })
                    
                    # Remove associated attendance records
                    self.db.attendance.delete_many({
                        'student_id': student_id,
                        'course_id': course_id
                    })
                    
                    # Remove associated engagement records
                    self.db.engagement.delete_many({
                        'student_id': student_id
                    })
                    
                    # Remove associated questions
                    self.db.questions.delete_many({
                        'student_id': student_id,
                        'course_id': course_id
                    })
                    
                    # Remove associated hand raises
                    self.db.hand_raises.delete_many({
                        'student_id': student_id,
                        'course_id': course_id
                    })
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error removing student and associated data: {str(e)}")
                    session.abort_transaction()
                    raise
        
    def log_hand_raise(self, student_id, course_id):
        """Log a hand raise event"""
        hand_raise = {
            'student_id': student_id,
            'course_id': course_id,
            'timestamp': datetime.utcnow()
        }
        return self.db.hand_raises.insert_one(hand_raise)
        
    def log_question(self, student_id, course_id, question_text, is_relevant, reason=""):
        """Log a question to the database"""
        try:
            # Create questions collection if it doesn't exist
            if 'questions' not in self.db.list_collection_names():
                self.db.create_collection('questions')
            
            # Insert question document
            result = self.db.questions.insert_one({
                'student_id': student_id,
                'course_id': course_id,
                'question_text': question_text,
                'is_relevant': is_relevant,
                'reason': reason,  # Add reason field
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error logging question: {str(e)}")
            raise
        
    def get_student_questions(self, student_id, course_id):
        """Get all questions asked by a student in a course"""
        try:
            logger.info(f"Fetching questions for student {student_id} in course {course_id}")
            questions = list(self.db.questions.find({
                'student_id': student_id,
                'course_id': course_id
            }).sort('timestamp', -1))
            logger.info(f"Found {len(questions)} questions")
            return questions
        except Exception as e:
            logger.error(f"Error fetching student questions: {str(e)}")
            return []
        
    def verify_questions_collection(self):
        """Verify the questions collection exists and is accessible"""
        try:
            # Check if collection exists
            collections = self.db.list_collection_names()
            logger.info(f"Available collections: {collections}")
            
            if 'questions' in collections:
                # Count documents
                count = self.db.questions.count_documents({})
                logger.info(f"Questions collection exists with {count} documents")
                
                # List some recent questions
                recent = list(self.db.questions.find().sort('timestamp', -1).limit(5))
                logger.info(f"Recent questions: {recent}")
                
                return True
            else:
                logger.error("Questions collection not found")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying questions collection: {str(e)}")
            return False
        
    def get_all_students(self):
        """Get all students across all courses"""
        return list(self.db.students.find())
        
    def get_course_questions(self, course_id):
        """Get all questions for a course"""
        try:
            # Get all registered students in the course first
            registered_students = {
                student['student_id']: student['name']
                for student in self.db.students.find({'course_id': course_id})
            }
            
            # Get questions only for registered students
            questions = list(self.db.questions.find({
                'course_id': course_id,
                'student_id': {'$in': list(registered_students.keys())}
            }).sort('timestamp', -1))
            
            # Add student names to questions
            for question in questions:
                question['student_name'] = registered_students.get(question['student_id'], 'Unknown Student')
            
            return questions
            
        except Exception as e:
            logger.error(f"Error getting course questions: {str(e)}")
            return []
        
    def get_student_attendance_rate(self, student_id, course_id):
        """Calculate student's attendance rate"""
        try:
            # Get all attendance records for the student
            total_records = self.db.attendance.count_documents({
                'student_id': student_id,
                'course_id': course_id
            })
            
            # Get present records
            present_records = self.db.attendance.count_documents({
                'student_id': student_id,
                'course_id': course_id,
                'status': 'Present'
            })
            
            # Calculate rate
            rate = (present_records / total_records * 100) if total_records > 0 else 0
            
            return {
                'rate': rate,
                'total_days': total_records,
                'days_present': present_records
            }
            
        except Exception as e:
            logger.error(f"Error calculating attendance rate: {str(e)}")
            return {'rate': 0, 'total_days': 0, 'days_present': 0}
        
    def update_student_name(self, student_id, course_id, new_name):
        """Update a student's name in the database"""
        try:
            # Update in students collection
            result = self.db.students.update_one(
                {
                    'student_id': student_id,
                    'course_id': course_id
                },
                {
                    '$set': {'name': new_name}
                }
            )
            
            if result.modified_count == 0:
                raise Exception("No student found with the given ID")
            
            # Update name in questions collection
            self.db.questions.update_many(
                {
                    'student_id': student_id,
                    'course_id': course_id
                },
                {
                    '$set': {'student_name': new_name}
                }
            )
            
            # Update name in attendance collection
            self.db.attendance.update_many(
                {
                    'student_id': student_id,
                    'course_id': course_id
                },
                {
                    '$set': {'student_name': new_name}
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating student name: {str(e)}")
            raise Exception(f"Failed to update student name: {str(e)}")
        
    def get_attendance_records(self, course_id, date_param):
        """Get attendance records for a specific course and date"""
        try:
            # Convert date to datetime at start of day if it's a date object
            if isinstance(date_param, date):
                date_start = datetime.combine(date_param, datetime.min.time())
            else:
                date_start = date_param.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get records for the entire day
            date_end = date_start + timedelta(days=1)
            
            records = self.db.attendance.find({
                'course_id': course_id,
                'date': {'$gte': date_start, '$lt': date_end}
            })
            return list(records)
            
        except Exception as e:
            logger.error(f"Error getting attendance records: {str(e)}")
            return []
        
    def get_student_by_id(self, student_id):
        """Get student by ID"""
        try:
            return self.db.students.find_one({'student_id': student_id})
        except Exception as e:
            logger.error(f"Error getting student by ID: {str(e)}")
            return None
        
    def get_student_hand_raises(self, student_id, course_id):
        """Get the number of hand raises for a student in a course"""
        try:
            return self.db.hand_raises.count_documents({
                'student_id': student_id,
                'course_id': course_id
            })
        except Exception as e:
            logger.error(f"Error getting hand raises count: {str(e)}")
            return 0
        
    def _setup_indexes(self):
        """Setup proper indexes for collections"""
        try:
            # Drop the old single-field index if it exists
            self.db.students.drop_index('student_id_1')
        except Exception:
            pass  # Index might not exist
        
        # Create the new compound index
        self.db.students.create_index(
            [('student_id', 1), ('course_id', 1)],
            unique=True,
            name='student_course_unique'
        )
        