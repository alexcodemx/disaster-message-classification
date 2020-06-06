from classificationapp import app
from model_scripts.model_data import tokenize

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()