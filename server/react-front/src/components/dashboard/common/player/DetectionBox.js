import React from "react";
import * as ReactDom from "react-dom";
import * as PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import Grow from "@material-ui/core/Grow";
import {stringToColour} from "../../../../utils/utils";

const styles = theme => ({
    selector: {
        position: "absolute",
        border: [["solid", theme.palette.secondary.main, "2px"]],
        boxSizing: "border-box",
    }
});

const DetectionBox = ({classes, boxes, on}) => {
    const selectors = (
        <Grow in={on}>
            <div style={{width: "100%", height: "100%"}}>
                {
                    boxes.map((box, index) =>
                        <div className={classes.selector}
                             key={index}
                             style={{
                                 borderColor: stringToColour(box.name),
                                 left: box.left,
                                 right: box.right,
                                 top: box.top,
                                 bottom: box.bottom,
                             }}/>
                    )
                }
            </div>
        </Grow>
    );

    return ReactDom.createPortal(selectors, document.querySelector("#hysia-detectionBox-container"));
};

DetectionBox.defaultProps = {
    boxes: [],
};

DetectionBox.propTypes = {
    classes: PropTypes.object.isRequired,
    boxes: PropTypes.arrayOf(
        PropTypes.shape({
            left: PropTypes.string.isRequired,
            right: PropTypes.string.isRequired,
            top: PropTypes.string.isRequired,
            bottom: PropTypes.string.isRequired,
        }).isRequired,
    ).isRequired,
    on: PropTypes.bool,
};

export default withStyles(styles)(DetectionBox);
