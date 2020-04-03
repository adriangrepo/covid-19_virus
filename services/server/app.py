# import time
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from dashboard.nb_mortality_rate import run_mcmc_model, fig_test
from dashboard.nb_covid19_growth import growth_workflow
import os
from bson.objectid import ObjectId
from datetime import datetime
from config.config import DB, CONF
from utils import setup_logging
import logging


'''
routes = [
    #Route('/users/{user_id:int}', user, methods=["GET", "POST"])
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/mortality", endpoint=mortality),
]
'''

logger = logging.getLogger(__name__)
setup_logging('app', log_level=CONF["log_level"])

app = Starlette(debug=CONF["mode"].get("debug"))

# A list of origins that should be permitted to make cross-origin requests
app.add_middleware(CORSMiddleware,
    allow_origins=[CONF["api"].get("cors_valid")],
)

def validate_object_id(id_: str):
    try:
        _id = ObjectId(id_)
    except Exception:
        if CONF["mode"].get("debug", False):
            logging.warning("Invalid Object ID")
        raise HTTPException(status_code=400)
    return _id

@app.route('/')
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
    uvicorn.run(app, host='0.0.0.0', port=8000)