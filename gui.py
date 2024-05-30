from flask import Flask, request, render_template
import search
import index

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def gui_search():
    token_positions = index.load_pickle_file('data/token_positions.pkl')
    doc_ids = index.load_pickle_file('data/doc_ids.pkl')
    pagerank = index.load_pickle_file('data/pagerank.pkl')


    #Get query from text box
    query = request.args.get('query')
    time_elapsed = 0
    result_count = 0

    postings = []
    if query:
        #Perform the search and return the results
        postings, time_elapsed = search.search(query, 'merged_index.txt', token_positions, doc_ids, pagerank)
        result_count = len(postings)
        time_elapsed = round(time_elapsed, 8)

    #Show the search form and the results (if any)
    return render_template('search.html', search_results=postings, query=query, time_elapsed=time_elapsed, result_count=result_count)

if __name__ == '__main__':
    app.run()