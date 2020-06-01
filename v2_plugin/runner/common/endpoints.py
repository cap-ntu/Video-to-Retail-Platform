import abc

import FastAPI


class BaseEndPoints(abc.ABC):

    name: str
    app = FastAPI(title=name, openapi_url='/openapi.json')

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @app.post('/predict')
    @abc.abstractmethod
    async def predict(self):
        raise NotImplementedError()
