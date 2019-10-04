import {VIDEO_DELETE_RECEIVE, VIDEO_DELETE_REQUEST} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";

const requestDelete = () => ({
    type: VIDEO_DELETE_REQUEST,
});

const receiveDelete_Success = json => ({
    type: VIDEO_DELETE_RECEIVE,
    status: "SUCCESS",
    time: json.time,
});

const receiveDelete_Failure = json => ({
    type: VIDEO_DELETE_RECEIVE,
    status: "FAILURE",
    reason: json.data,
    time: json.time,
});

export const VIDEO_delete = (id, successCallback) => dispatch => {

    xmlHttpRequest(dispatch, "DELETE", {
        url: `/restapi/videos/${id}/`,
        request: requestDelete,
        receiveSuccess: json => {
            successCallback();
            return receiveDelete_Success(json);
        },
        receiveFailure: receiveDelete_Failure
    });
};