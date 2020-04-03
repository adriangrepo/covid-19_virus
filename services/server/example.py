# import time
from starlette.applications import Starlette, Route
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from scripts.mortality_rate import run_mcmc_model

app = Starlette(debug=True)

# A list of origins that should be permitted to make cross-origin requests
app.add_middleware(CORSMiddleware,
    allow_origins=['http://localhost:8080'],
)

@app.route('/')
async def homepage(request):
    # time.sleep(10)
    return JSONResponse({'hello': 'world'})

@app.route('/mortality')
async def mortality(request):
    reported_mortality_rate, mortality_rate = run_mcmc_model()
    return JSONResponse({'reported mortality rate': f'{reported_mortality_rate}'})

routes = [
    #Route('/users/{user_id:int}', user, methods=["GET", "POST"])
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/mortality", endpoint=mortality),
]

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)