import React from "react";
import path from "path";

/**
 * route wrapper function. It dynamically determines current app root. Need to be executed only when its parent
 * Component is mounted.
 * {@code
 * <Switch>
 *     {RouteWrapper(props)}
 * </Switch>
 * }
 * Cannot statically loaded: {@code const router = ({RouteWrapper(props)});}, the generated Router components will not
 * show correct path.
 * @param Component react-router Component, one of <Router/>, <Redirect/>
 * @param Compo rendered Component when the path is matched
 * @param path: current url/path
 * @param rest other props, {@see RouteWrapper.propTypes}
 * @returns {*}
 * @constructor null
 */
export default function wrapper(Compo, {path: _path, from, to, component: Component, ...rest}) {
    function render(props) {
        window.appRoot = props.match.path;
        return Component ? <Component {...props}/> : null;
    }

    function refToAbsolute(_path) {
        return _path ? path.join(window.appRoot || "/", _path) : undefined;
    }

    _path = refToAbsolute(_path);
    from = refToAbsolute(from);
    to = refToAbsolute(to);
    return <Compo {...rest} path={_path} from={from} to={to} render={render}/>
}
