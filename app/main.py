from fastapi import FastAPI
from .modules.search import *
from .modules.models import *
from tqdm import tqdm
from transliterate import translit, get_available_language_codes

DATASET_PATH = 'app/additional_data/building_20230808.csv'
tqdm.pandas()
TRAINED_MODEL_PATH = 'app/additional_data/rubert_tiny2_negs_2epoch.pt'


globals = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
     
    global globals

    DB = proceed_data(DATASET_PATH)

    uniq_adresses = DB['full_address'].unique()
    proceed_uniq_adresses = list(map(convert_string, uniq_adresses))
    uniq_words = get_uWords(proceed_uniq_adresses)

    lev_word_tokenizer = {word: index for index, word in enumerate(uniq_words)}

    DB['tokenize_addresses'] =  DB['proceed_target'].apply(lambda x: words2nums(lev_word_tokenizer, x))
    print(DB.head())

    default_bert = BertSearcher()

    trained_bert = BertSearcher()
    # trained_bert.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu')))
    globals = {
        "DB": DB,
        "trained_bert": trained_bert,
        "default_bert": default_bert,
        "lev_word_tokenizer": lev_word_tokenizer,
        "uniq_words": uniq_words
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
async def query(query: Query = None):
    queries = [translit(i, 'ru') for i in query.objects]
    success, response_list = make_response(queries, globals)

    response = {
        "success": success,
        "query": queries,
        "result": response_list
    }
    return response
