import React from "react";
import {connect} from "react-redux"
import Snackbar from "../../common/Snackbar";

const mapStateToProps = state => ({
    newState: state.user.userNew,
    updateState: state.user.userUpdate,
    deleteState: state.user.userDelete,
});

const MessageBars = ({newState, updateState, deleteState}) => (
    <React.Fragment>
        <Snackbar state={newState} message={{success: "Create user success"}}/>
        <Snackbar state={updateState} message={{success: "Update user success"}}/>
        <Snackbar state={deleteState} message={{success: "Delete user success"}}/>
    </React.Fragment>
);

export default connect(mapStateToProps)(MessageBars);
