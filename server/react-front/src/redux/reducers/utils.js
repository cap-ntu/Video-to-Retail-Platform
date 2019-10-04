export function asyncReducer(state, action, {request, receive}, mappings = {}) {
    switch (action.type) {
        case request:
            return {
                ...state,
                state: "REQUEST",
            };
        case receive:
            if (action.status === "SUCCESS")
                return {
                    ...state,
                    state: "SUCCESS",
                    ...mappings,
                };
            return {
                ...state,
                state: "FAILURE",
                reason: action.reason,
            };
        default:
            return {...state};
    }
}
