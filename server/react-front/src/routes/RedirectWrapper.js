import Redirect from "react-router/Redirect";
import * as PropTypes from "prop-types";
import wrapper from "./wrapper";

function RedirectWrapper({from, to, ...rest}) {
    return wrapper(Redirect, {from, to, ...rest});
}

RedirectWrapper.propTypes = {
    from: PropTypes.string,
    to: PropTypes.string,
    push: PropTypes.bool,
    exact: PropTypes.bool,
    strict: PropTypes.bool,
};

export default RedirectWrapper;
