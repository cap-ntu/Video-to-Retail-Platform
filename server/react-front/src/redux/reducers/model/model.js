import {
    MODEL_DELETE_RECEIVE,
    MODEL_DELETE_REQUEST,
    MODEL_GET_DEFAULT_LIST_RECEIVE,
    MODEL_GET_DEFAULT_LIST_REQUEST,
    MODEL_GET_LIST_RECEIVE,
    MODEL_GET_LIST_REQUEST,
    MODEL_GET_SINGLE_RECEIVE,
    MODEL_GET_SINGLE_REQUEST,
    MODEL_POST_RECEIVE,
    MODEL_POST_REQUEST
} from "../../../constants/actionTypes";
import {asyncReducer} from "../utils";

const initState = {
    models: {
        state: "INIT",
        time: null,
        reason: "",
        models: {},
    },
    defaultModels: {
        state: "INIT",
        time: null,
        reason: "",
        models: {},
    },
    singleModel: {
        state: "INIT",
        time: null,
        reason: "",
        model: {},
    },
    newModel: {
        state: "INIT",
        reason: "",
        time: null,
    },
    deleteModel: {
        state: "INIT",
        reason: "",
        time: null,
    }
};

export default function model(state = initState, action) {
    switch (action.type) {
        case MODEL_GET_LIST_REQUEST:
        case MODEL_GET_LIST_RECEIVE:
            return {
                ...state,
                models: asyncReducer(state.models, action,
                    {
                        request: MODEL_GET_LIST_REQUEST,
                        receive: MODEL_GET_LIST_RECEIVE,
                    },
                    {models: action.models}
                )
            };
        case MODEL_GET_DEFAULT_LIST_REQUEST:
        case MODEL_GET_DEFAULT_LIST_RECEIVE:
            return {
                ...state,
                defaultModels: asyncReducer(state.defaultModels, action,
                    {
                        request: MODEL_GET_DEFAULT_LIST_REQUEST,
                        receive: MODEL_GET_DEFAULT_LIST_RECEIVE,
                    },
                    {models: action.models})
            };
        case MODEL_GET_SINGLE_REQUEST:
        case MODEL_GET_SINGLE_RECEIVE:
            return {
                ...state,
                singleModel: asyncReducer(state.singleModel, action,
                    {
                        request: MODEL_GET_SINGLE_REQUEST,
                        receive: MODEL_GET_SINGLE_RECEIVE,
                    },
                    {model: action.model})
            };
        case MODEL_POST_REQUEST:
        case MODEL_POST_RECEIVE:
            return {
                ...state,
                newModel: asyncReducer(state.newModel, action,
                    {
                        request: MODEL_POST_REQUEST,
                        receive: MODEL_POST_RECEIVE,
                    })
            };
        case MODEL_DELETE_REQUEST:
        case MODEL_DELETE_RECEIVE:
            return {
                ...state,
                deleteModel: asyncReducer(state.deleteModel, action,
                    {
                        request: MODEL_DELETE_REQUEST,
                        receive: MODEL_DELETE_RECEIVE,
                    })
            };
        default:
            return {...state};
    }
}
