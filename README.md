SEARCH ENGINE

This is a simple search engine, allowing you to input queries to search for relevant documents within the UCI domain.

STEP 1:
Install dependencies. Run the following command in terminal:
pip install -r requirements.txt

STEP 2:
For first time users, you must run the file "index.py" to create the necessary files for searching. This should take 45 min to 1.5 hrs, so just wait until it says everything is complete in the terminal to start searching. 
Use the following terminal command to run index.py:
python -u index.py

STEP 3:
Run "gui.py" and open "http://127.0.0.1:5000" on your machine to search through the web interface. The interface consists of a simple search box. Search whatever you want and you will be met with a list of links ranked from most relevant to least relevant.
Use the following terminal command to run gui.py:
python -u gui.py

Alternatively, run "search.py" to use a terminal search interface. Simply run "search.py" and enter your query into the prompted input.
Run:
python -u search.py