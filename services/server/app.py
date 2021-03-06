# import time
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn
from dashboard.nb_mortality_rate import run_mcmc_model, fig_test
from dashboard.nb_covid19_growth import growth_workflow
import os
from bson.objectid import ObjectId
from datetime import datetime
from db import DB
from utils import setup_logging
import logging
from config.server_config import DEBUG_MODE, API_CORS_VALID, ASGI_HOST, ASGI_PORT


'''
routes = [
    #Route('/users/{user_id:int}', user, methods=["GET", "POST"])
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/mortality", endpoint=mortality),
]
'''

logger = logging.getLogger(__name__)
setup_logging()

templates = Jinja2Templates(directory='templates')

app = Starlette(debug=DEBUG_MODE)

# A list of origins that should be permitted to make cross-origin requests
app.add_middleware(CORSMiddleware,
    allow_origins=[API_CORS_VALID],
)

logger.info(f'app started, debug mode: {DEBUG_MODE}')


def validate_object_id(id_: str):
    try:
        _id = ObjectId(id_)
    except Exception:
        if DEBUG_MODE == False:
            logging.warning("Invalid Object ID")
        raise HTTPException(status_code=400)
    return _id

@app.route('/')
async def homepage(request):
    logger.debug('route / ')
    template = "index.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)    

@app.route('/hello')
async def homepage(request):
    # time.sleep(10)
    return JSONResponse({'hello': 'backend response'})

@app.route('/confirmed')
async def confirmed(request):
    confirmed()
    return JSONResponse({'growth': 'completed'})

@app.route('/mortality_model')
async def mortality_model(request):
    reported_mortality_rate, mortality_rate = run_mcmc_model()
    return JSONResponse({'reported mortality rate': f'{reported_mortality_rate}'})

@app.route('/infected_estimate')
async def infected_estimate(request):
    countries = request.path_params['countries']
    countries_infected_ts = await DB.csse_estimated_infected_ts.find({"location": {"$in": countries}})
    return JSONResponse({'infected_estimate_ts': countries_infected_ts})

@app.route('/infected_estimate_stats')
async def infected_estimate_stats(request):
    countries = request.path_params['countries']
    countries_infected_stats = await DB.csse_estimated_infected_latest_stats.find({"location": {"$in": countries}})
    return JSONResponse({'infected_estimate_stats': countries_infected_stats})

@app.route('/test')
async def test(request):
    print('route/image')
    pth=fig_test()
    return JSONResponse({'path': f'{pth}'})



if __name__ == '__main__':
    uvicorn.run(app, host=ASGI_HOST, port=ASGI_PORT)