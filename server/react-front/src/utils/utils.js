import assert from "assert";

export function isEmpty(object) {
    return Object.keys(object).length === 0;
}

export function secondToHMS(s) {
    const day = Math.floor(s / 86400);
    const str = new Date(s * 1000).toISOString().substr(11, 8);
    return day > 0 ? `${day}:${str}` : str.startsWith("00:") ? str.substr(3) : str;
}

/**
 * @author Joe Freeman
 * @see https://stackoverflow.com/questions/3426404/create-a-hexadecimal-colour-based-on-a-string-with-javascript
 * @param str
 * @returns {string}
 */
export function stringToColour(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    let colour = '#';
    for (let i = 0; i < 3; i++) {
        const value = (hash >> (i * 8)) & 0xFF;
        colour += ('00' + value.toString(16)).substr(-2);
    }
    return colour;
}

/**
 *
 * @param _this parent dom
 * @param setState parent setState function
 * @param allow permitted key
 * @returns {setState} strict setState
 */
export function injectParentState(_this, setState, allow) {

    return state => {
        if (typeof setState === "function" || !allow)    // bypass allow key checking
            setState.bind(_this)(state);
        else {
            Object.keys(state).map(key => {
                assert(allow.includes(key), `${key} is not a permitted field to set.`);
                setState.bind(_this)(state);
            })
        }
    }
}

export function emptyFunction() {
}

/**
 * String hash
 * @see{https://stackoverflow.com/a/7616484}
 * @return {number} hash code of a string
 */
String.prototype.hashCode = function () {
    var hash = 0, i, chr;
    if (this.length === 0) return hash;
    for (i = 0; i < this.length; i++) {
        chr = this.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};

export function getCookie(name) {
    const value = '; ' + document.cookie;
    const parts = value.split('; ' + name + '=');
    if (parts.length === 2)
        return parts.pop().split(';').shift()
}

export function deleteCookie(name) {
    const value = "; " + document.cookie;
    const parts = value.split("; " + name + '=');
    if (parts.length === 2)
        document.cookie = parts[0] + parts[1].slice((parts[1] + ";").indexOf(";"))
}
