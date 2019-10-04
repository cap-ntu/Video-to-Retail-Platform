import React from "react";
import {connect} from "react-redux"
import Snackbar from "../../common/Snackbar";

const mapStateToProps = state => ({
    deleteState: state.model.deleteModel,
    newState: state.model.newModel,
});

const MessageBars = ({newState, deleteState}) => (
    <React.Fragment>
        <Snackbar state={newState} message={{success: "Create model succeed"}}/>
        <Snackbar state={deleteState} message={{success: "Delete model succeed"}}/>
    </React.Fragment>
);

export default connect(mapStateToProps)(MessageBars);
