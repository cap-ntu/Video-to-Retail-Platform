import {USER_UPDATE_RECEIVE, USER_UPDATE_REQUEST} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestUpdate = () => ({
    type: USER_UPDATE_REQUEST,
});

const receiveUpdate_Success = json => ({
    type: USER_UPDATE_RECEIVE,
    status: 'SUCCESS',
    time: json.time,
});

const receiveUpdate_Failure = json => ({
    type: USER_UPDATE_RECEIVE,
    status: 'FAILURE',
    reason: json.data,
});

export const USER_update = ({id, firstName, lastName, email, password, username, groups, accountType, domain, status},
                            successCallback = emptyFunction) => dispatch => {

    const formData = new FormData();
    formData.append("id", id);
    formData.append("first_name", firstName);
    formData.append("last_name", lastName);
    formData.append("email", email);
    groups.forEach(group => formData.append("groups[]", group));
    if (accountType === "Administrator") formData.append("is_superuser", "true");
    if (domain === "Staff") formData.append("is_staff", "true");
    if (status === "Activated") formData.append("is_active", "true");

    xmlHttpRequest(dispatch, "PATCH", {
        url: `/restapi/users/${id}/`,
        request: requestUpdate,
        receiveSuccess: json => {
            successCallback();
            return receiveUpdate_Success(json);
        },
        receiveFailure: receiveUpdate_Failure,
        formData,
    });
};