import {getCookie} from "./utils";

export function formatParams(params) {
    return "?" + Object
        .keys(params)
        .map(function (key) {
            return key + "=" + encodeURIComponent(params[key])
        })
        .join("&")
}

export function xmlHttpRequest(dispatch, method, {
    url, jsonDecoder, request, receiveSuccess, receiveFailure, progressListener, formData = null
}) {

    switch (method.toString().toUpperCase()) {
        case "GET":
            return get(dispatch, url, jsonDecoder, request, receiveSuccess, receiveFailure, formData);
        case "POST":
            return post(dispatch, url, jsonDecoder, request, receiveSuccess, receiveFailure, progressListener, formData);
        case "PATCH":
            return patch(dispatch, url, request, receiveSuccess, receiveFailure, formData);
        case "DELETE":
            return del(dispatch, url, request, receiveSuccess, receiveFailure, formData);
        default:
            return;
    }
}

function get(dispatch, url, jsonDecoder = json => json, request, receiveSuccess, receiveFailure, formData) {

    function onStateChange() {
        if (xmlReq.readyState === 4) {
            if (xmlReq.status >= 200 && xmlReq.status <= 300) {
                try {
                    const json = JSON.parse(xmlReq.responseText);
                    const data = jsonDecoder(json.data);
                    dispatch(receiveSuccess({...json, data: data}));
                } catch (e) {
                    dispatch(receiveFailure({data: "Error in mapping response json to list"}));
                }
            } else
                dispatch(receiveFailure({data: `Other error: ${xmlReq.statusText}`}));
        }
    }

    // GET start
    dispatch(request());

    const xmlReq = new XMLHttpRequest();
    xmlReq.open('GET', url, true);
    xmlReq.onreadystatechange = onStateChange;
    xmlReq.send(formData);

    return xmlReq;
}

function post(dispatch, url, jsonDecoder = json => json, request, receiveSuccess, receiveFailure, progressListener, formData) {

    function onLoad() {
        if (xhr.status >= 200 && xhr.status < 300) {
            const json = JSON.parse(xhr.responseText);
            const data = jsonDecoder(json.data);
            dispatch(receiveSuccess({...json, data: data}));
        } else
            dispatch(receiveFailure({data: `${xhr.status} (${xhr.statusText})`}))
    }

    const onProgress = progressListener ? event =>
        progressListener(event.lengthComputable, event.loaded, event.total) : null;

    // POST start
    dispatch(request());

    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.withCredentials = true;
    xhr.onload = onLoad;
    xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
    xhr.onprogress = onProgress;
    xhr.send(formData);

    return xhr;
}

function patch(dispatch, url, request, receiveSuccess, receiveFailure, formData) {

    function onLoad() {
        if (xmlReq.status >= 200 && xmlReq.status < 300) {
            window.temp1 = xmlReq.responseText;
            console.log(receiveSuccess(JSON.parse(window.temp1)));
            dispatch(receiveSuccess(JSON.parse(xmlReq.responseText)));
        } else
            dispatch(receiveFailure({data: `${xmlReq.status} Error: ${xmlReq.statusText}`}));
    }

    // UPDATE start
    dispatch(request());

    const xmlReq = new XMLHttpRequest();
    xmlReq.open("PATCH", url, true);
    xmlReq.withCredentials = true;
    xmlReq.onload = onLoad;
    xmlReq.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
    xmlReq.send(formData);

    return xmlReq;
}

function del(dispatch, url, request, receiveSuccess, receiveFailure, formData) {

    function onLoad() {
        if (xmlReq.status >= 200 && xmlReq.status < 300)
            dispatch(receiveSuccess(JSON.parse(xmlReq.responseText)));
        else
            dispatch(receiveFailure({data: `${xmlReq.status} Error: ${xmlReq.statusText}`}));
    }

    // DELETE start
    dispatch(request());

    const xmlReq = new XMLHttpRequest();
    xmlReq.open("DELETE", url, true);
    xmlReq.withCredentials = true;
    xmlReq.onload = onLoad;
    xmlReq.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
    xmlReq.send(formData);

    return xmlReq;
}
