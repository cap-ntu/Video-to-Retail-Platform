import {MODEL_POST_RECEIVE, MODEL_POST_REQUEST,} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestCreate = () => ({
    type: MODEL_POST_REQUEST,
});

const receiveCreate_Success = json => ({
    type: MODEL_POST_RECEIVE,
    status: 'SUCCESS',
    time: json.time,
});

const receiveCreate_Failure = json => ({
    type: MODEL_POST_RECEIVE,
    status: 'FAILURE',
    reason: json.data,
});

export const MODEL_create = ({type, name, model, label}, successCallback = emptyFunction) => dispatch => {

    const formData = new FormData();
    formData.append("type", type);
    formData.append("name", name);
    if (model) formData.append("model", model, model.name);
    if (label) formData.append("label", label, label.name);

    xmlHttpRequest(dispatch, "POST", {
        url: '/restapi/dlmodels/',
        request: requestCreate,
        receiveSuccess: json => {
            successCallback();
            return receiveCreate_Success(json);
        },
        receiveFailure: receiveCreate_Failure,
        formData,
    });
};
