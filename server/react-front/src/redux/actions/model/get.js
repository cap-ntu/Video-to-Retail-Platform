import {
    MODEL_GET_DEFAULT_LIST_RECEIVE,
    MODEL_GET_DEFAULT_LIST_REQUEST,
    MODEL_GET_LIST_RECEIVE,
    MODEL_GET_LIST_REQUEST,
    MODEL_GET_SINGLE_RECEIVE,
    MODEL_GET_SINGLE_REQUEST,
} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {decodeModelListJson, decodeSingleModelJson} from "../../../utils/jsonDecoder";
import {emptyFunction} from "../../../utils/utils";

const requestGetList = () => ({
    type: MODEL_GET_LIST_REQUEST,
});

const receiveGetList_Success = json => ({
    type: MODEL_GET_LIST_RECEIVE,
    status: "SUCCESS",
    models: json.data,
    time: json.time,
});

const receiveGetList_Failure = json => ({
    type: MODEL_GET_LIST_RECEIVE,
    status: "FAILURE",
    reason: json.data,
});

export const MODEL_getList = (successCallback = emptyFunction) => dispatch => {

    // const data = require("../../../asset/dlmodellist.json");

    xmlHttpRequest(dispatch, "GET", {
        url: '/restapi/dlmodels/',
        jsonDecoder: decodeModelListJson,
        request: requestGetList,
        receiveSuccess: json => {
            successCallback();
            return receiveGetList_Success(json);
        },
        receiveFailure: receiveGetList_Failure,
    });

    // successCallback();
    // dispatch(receiveGetList_Success({data: decodeModelListJson(data.data)}));
};

const requestGetDefaultList = () => ({
    type: MODEL_GET_DEFAULT_LIST_REQUEST,
});

const receiveGetDefaultList_Success = json => ({
    type: MODEL_GET_DEFAULT_LIST_RECEIVE,
    status: "SUCCESS",
    models: json.data,
    time: json.time,
});

const receiveGetDefaultList_Failure = json => ({
    type: MODEL_GET_DEFAULT_LIST_RECEIVE,
    status: "FAILURE",
    reason: json.data,
});

export const MODEL_getDefaultList = (successCallback = emptyFunction) => dispatch => {

    // const data = require("../../../asset/defaultmodels.json");
    xmlHttpRequest(dispatch, "GET", {
        url: '/restapi/defaultdlmodels/',
        jsonDecoder: decodeModelListJson,
        request: requestGetDefaultList,
        receiveSuccess: json => {
            successCallback();
            return receiveGetDefaultList_Success(json);
        },
        receiveFailure: receiveGetDefaultList_Failure
    });

    // successCallback();
    // dispatch(receiveGetDefaultList_Success({data: decodeModelListJson(data.data)}));
};

const requestGetSingle = () => ({
    type: MODEL_GET_SINGLE_REQUEST,
});

const requestGetSingle_Success = json => ({
    type: MODEL_GET_SINGLE_RECEIVE,
    status: 'SUCCESS',
    model: json.data,
    time: json.time,
});

const requestGetSingle_Failure = json => ({
    type: MODEL_GET_SINGLE_RECEIVE,
    status: 'FAILURE',
    reason: json.data,
});

export const MODEL_getSingle = id => dispatch => {

    const formData = new FormData();
    formData.append("id", id);

    xmlHttpRequest(dispatch, "GET", {
        url: `/restapi/dlmodels/${id}/`,
        jsonDecoder: decodeSingleModelJson,
        request: requestGetSingle,
        receiveSuccess: requestGetSingle_Success,
        receiveFailure: requestGetSingle_Failure,
        formData,
    });
};
