import {USER_POST_RECEIVE, USER_POST_REQUEST,} from "../../../constants/actionTypes";
import {xmlHttpRequest} from "../../../utils/httpRequest";
import {emptyFunction} from "../../../utils/utils";

const requestCreate = () => ({
    type: USER_POST_REQUEST,
});

const receiveCreate_Success = json => ({
    type: USER_POST_RECEIVE,
    status: 'SUCCESS',
    time: json.time,
});

const receiveCreate_Failure = json => ({
    type: USER_POST_RECEIVE,
    status: 'FAILURE',
    reason: json.data,
});

export const USER_create = ({username, password, firstName, lastName, email, groups, accountType, domain, status},
                            successCallback = emptyFunction) => dispatch => {

    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);
    formData.append("first_name", firstName);
    formData.append("last_name", lastName);
    formData.append("email", email);
    groups.forEach(group => formData.append("groups", group));
    if (accountType === "Administrator") formData.append("is_superuser", "true");
    if (domain === "Staff") formData.append("is_staff", "true");
    if (status === "Activated") formData.append("is_active", "true");

    xmlHttpRequest(dispatch, "POST", {
        url: '/restapi/users/',
        request: requestCreate,
        receiveSuccess: json => {
            successCallback();
            return receiveCreate_Success(json);
        },
        receiveFailure: receiveCreate_Failure,
        formData,
    });
};
