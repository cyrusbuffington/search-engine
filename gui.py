from flask import Flask, request, render_template
import search
import index

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def gui_search():
    token_positions = index.load_pickle_file('data/token_positions.pkl')
    doc_ids = index.load_pickle_file('data/doc_ids.pkl')


    #Get query from text box
    query = request.args.get('query')

    postings = []
    if query:
        # Perform the search and return the results
        postings = search.search(query, 'merged_index.txt', token_positions, doc_ids)[:10]

    #Show the search form and the results (if any)
    return render_template('search.html', search_results=postings, query=query)

# main driver function
if __name__ == '__main__':
    app.run()