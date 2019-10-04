import React from "react";
import * as PropTypes from "prop-types";
import Collapse from "@material-ui/core/Collapse";
import Grow from "@material-ui/core/Grow";

const CollapseGrow = ({in: _in, children, style, ...rest}) => (
    <Collapse in={_in} style={style}>
        <Grow in={_in} {...rest}>
            {children}
        </Grow>
    </Collapse>
);

CollapseGrow.propTypes = {
    in: PropTypes.bool,
    style: PropTypes.object,
    onEnter: PropTypes.func,
    onExited: PropTypes.func,
};

export default CollapseGrow;
