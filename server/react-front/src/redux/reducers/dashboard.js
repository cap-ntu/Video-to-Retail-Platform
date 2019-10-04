import {DSH_RECEIVE_STATISTICS, DSH_REQUEST_STATISTICS} from "../../constants/actionTypes";

const initState = {
    CPU: [],
    Memory: [],
    Disk: [],
    Network: [],
    GPU: [],
    summary: {},
};

function calculateOverAll(dataArray) {
    try {
        const resourceInstance = dataArray.length;
        return {
            id: "Overall",
            data: dataArray.map(resource => resource.data).reduce((x, y) => x.map((e, i) => ({...e, y: e.y + y[i].y})))
                .map(resource => ({...resource, y: resource.y / resourceInstance}))
        };
    } catch (e) {
    }
    return {id: "Overall", data: []}
}

function resourceClassifier(data, names=[]) {
    let resourcesData = {"summary": {}};
    names.forEach(name => {
        let resourceData = data.filter(resource => resource.id.match(name.toString()) !== null);
        resourcesData[name] = resourceData;
        resourcesData.summary[name] = calculateOverAll(resourceData);
    });
    return resourcesData;
}

function dashboard(state = initState, action) {
    switch (action.type) {
        case DSH_REQUEST_STATISTICS:
            return state;
        case DSH_RECEIVE_STATISTICS:
            if (action.status === "SUCCESS") {
                return {
                    ...state,
                    ...resourceClassifier(action.dataArray, ["CPU", "Memory", "Disk", "Network", "GPU"])
                }
            }
            console.log(action.error);
            return state;
        default:
            return state;
    }
}

export default dashboard;
