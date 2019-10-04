import {VIDEO_UPDATE_RECEIVE, VIDEO_UPDATE_REQUEST} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestUpdate = () => ({
    type: VIDEO_UPDATE_REQUEST,
});

const receiveUpdate_Success = json => ({
    type: VIDEO_UPDATE_RECEIVE,
    status: 'SUCCESS',
    time: json.time,
});

const receiveUpdate_Failure = json => ({
    type: VIDEO_UPDATE_RECEIVE,
    status: 'FAILURE',
    reason: json.data,
});

// request for process
export const VIDEO_update = (id, successCallback = emptyFunction) => dispatch => {

    const formData = new FormData();
    formData.append("id", id);

    xmlHttpRequest(dispatch, "PATCH", {
        url: `/restapi/videos/${id}/`,
        request: requestUpdate,
        receiveSuccess: json => {
            successCallback();
            return receiveUpdate_Success(json);
        },
        receiveFailure: receiveUpdate_Failure,
        formData,
    });
};