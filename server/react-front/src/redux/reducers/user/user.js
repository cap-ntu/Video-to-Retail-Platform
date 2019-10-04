import {
    USER_DELETE_RECEIVE,
    USER_DELETE_REQUEST,
    USER_GET_LIST_RECEIVE,
    USER_GET_LIST_REQUEST,
    USER_POST_RECEIVE,
    USER_POST_REQUEST,
    USER_UPDATE_RECEIVE,
    USER_UPDATE_REQUEST
} from "../../../constants/actionTypes";
import {asyncReducer} from "../utils";

const initState = {
    userList: {
        state: "INIT",
        time: null,
        reason: "",
        users: [],
        nameList: {},
    },
    userNew: {
        state: "INIT",
        time: null,
        reason: "",
    },
    userUpdate: {
        state: "INIT",
        time: null,
        reason: "",
    },
    userDelete: {
        state: "INIT",
        time: null,
        reason: "",
    },
};

export default function (state = initState, action) {
    switch (action.type) {
        case USER_GET_LIST_REQUEST:
        case USER_GET_LIST_RECEIVE:
            return {
                ...state,
                userList: asyncReducer(state.userList, action,
                    {
                        request: USER_GET_LIST_REQUEST,
                        receive: USER_GET_LIST_RECEIVE
                    },
                    {
                        users: action.users,
                        nameList: action.nameList
                    })
            };
        case USER_POST_REQUEST:
        case USER_POST_RECEIVE:
            return {
                ...state,
                userNew: asyncReducer(state.userNew, action,
                    {
                        request: USER_POST_REQUEST,
                        receive: USER_POST_RECEIVE,
                    })
            };
        case USER_UPDATE_REQUEST:
        case USER_UPDATE_RECEIVE:
            return {
                ...state,
                userUpdate: asyncReducer(state.userUpdate, action,
                    {
                        request: USER_UPDATE_REQUEST,
                        receive: USER_UPDATE_RECEIVE,
                    })
            };
        case USER_DELETE_REQUEST:
        case USER_DELETE_RECEIVE:
            return {
                ...state,
                userDelete: asyncReducer(state.userDelete, action,
                    {
                        request: USER_DELETE_REQUEST,
                        receive: USER_DELETE_RECEIVE,
                    })
            };
        default:
            return {...state};
    }
}
