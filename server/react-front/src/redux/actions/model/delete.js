import {MODEL_DELETE_RECEIVE, MODEL_DELETE_REQUEST} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestDelete = () => ({
    type: MODEL_DELETE_REQUEST,
});

const receiveDelete_Success = json => ({
    type: MODEL_DELETE_RECEIVE,
    status: "SUCCESS",
    time: json.time,
});

const receiveDelete_Failure = json => ({
    type: MODEL_DELETE_RECEIVE,
    status: "FAILURE",
    reason: json.data,
});

export const MODEL_delete = (id, callback = emptyFunction) => dispatch => {
    xmlHttpRequest(dispatch, "DELETE", {
        url: `/restapi/dlmodels/${id}/`,
        request: requestDelete,
        receiveSuccess: json => {
            callback();
            return receiveDelete_Success(json);
        },
        receiveFailure: receiveDelete_Failure,
    });
};