import * as PropTypes from "prop-types";
import Route from "react-router/Route";
import wrapper from "./wrapper";

function RouteWrapper({path, ...rest}) {
    return wrapper(Route, {path: path, ...rest});
}

RouteWrapper.propTypes = {
    path: PropTypes.string,
    exact: PropTypes.bool,
    strict: PropTypes.bool,
    sensitive: PropTypes.bool,
    component: PropTypes.func,
    render: PropTypes.func,
};

export default RouteWrapper;
