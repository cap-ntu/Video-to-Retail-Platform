import {USER_DELETE_RECEIVE, USER_DELETE_REQUEST} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestDelete = () => ({
    type: USER_DELETE_REQUEST,
});

const receiveDelete_Success = json => ({
    type: USER_DELETE_RECEIVE,
    status: "SUCCESS",
    time: json.time,
});

const receiveDelete_Failure = json => ({
    type: USER_DELETE_RECEIVE,
    status: "FAILURE",
    reason: json.data,
});

export const USER_delete = (id, successCallback = emptyFunction) => dispatch => {
    xmlHttpRequest(dispatch, "DELETE", {
        url: `/restapi/users/${id}/`,
        request: requestDelete,
        receiveSuccess: json => {
            successCallback();
            return receiveDelete_Success(json);
        },
        receiveFailure: receiveDelete_Failure,
    });
};