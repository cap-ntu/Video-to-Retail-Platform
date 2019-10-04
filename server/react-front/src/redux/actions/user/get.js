import {
    USER_GET_LIST_RECEIVE,
    USER_GET_LIST_REQUEST,
    USER_LOGOUT_RECEIVE,
    USER_LOGOUT_REQUEST
} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {decodeUserListJson} from "../../../utils/jsonDecoder";
import {emptyFunction} from "../../../utils/utils";

const requestGetList = () => ({
    type: USER_GET_LIST_REQUEST,
});

function usersToNameList(data) {
    const nameList = {};

    data.map(user => user.username).forEach(user => {
        let cat = user[0].toUpperCase();
        cat = cat.match(/[A-Z]/) ? cat : (cat.match(/[0-9]/) ? "#" : "@");
        nameList.hasOwnProperty(cat) ? nameList[cat].push(user) : nameList[cat] = [user];
    });

    return nameList;
}

const receiveGetList_Success = (json) => ({
    type: USER_GET_LIST_RECEIVE,
    status: "SUCCESS",
    users: json.data,
    nameList: usersToNameList(json.data),
    time: json.time,
});

const receiveGetList_Failure = (json) => ({
    type: USER_GET_LIST_RECEIVE,
    status: "FAILURE",
    reason: json.data,
});

export const USER_getList = () => dispatch => {
    xmlHttpRequest(dispatch, "GET", {
        url: '/restapi/users/',
        jsonDecoder: decodeUserListJson,
        request: requestGetList,
        receiveSuccess: receiveGetList_Success,
        receiveFailure: receiveGetList_Failure,
    });
};

const requestLogout = () => ({
    type: USER_LOGOUT_REQUEST,
});

const receiveLogout_Success = () => ({
    type: USER_LOGOUT_RECEIVE,
    status: "SUCCESS",
});

const receiveLogout_Failure = json => ({
    type: USER_LOGOUT_RECEIVE,
    status: "SUCCESS",
    reason: json.data,
});

export const USER_logout = (successCallback = emptyFunction) => dispatch => {

    xmlHttpRequest(dispatch, "GET", {
        url: '/restapi/logout/',
        request: requestLogout,
        receiveSuccess: json => {
            successCallback();
            return receiveLogout_Success(json);
        },
        receiveFailure: receiveLogout_Failure,
    });
};
