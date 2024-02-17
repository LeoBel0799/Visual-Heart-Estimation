from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)
    current_directory = path.dirname(path.abspath(__file__))
    project_directory = path.dirname(current_directory)
    shape_predictor_path = path.join(project_directory, r'shape_predictor_68_face_landmarks_GTX.dat')
    face_detector_path = path.join(project_directory, r'mmod_human_face_detector.dat')
    max_min_path = path.join(project_directory, r'min_max_values_vit.txt')
    model_path = path.join(project_directory,r'VIT_jit.pt')

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Video, File

    with app.app_context():
        db.create_all()

        files_to_check = [
            shape_predictor_path,
            face_detector_path,
            max_min_path,
        ]

        for file_path in files_to_check:
            if not File.query.filter_by(filename=file_path).first():
                with open(file_path, 'rb') as file:
                    content = file.read()
                    new_file = File(filename=file_path, content=content)
                    db.session.add(new_file)
                db.session.commit()
            db.session.commit()
        else:
            pass


    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')