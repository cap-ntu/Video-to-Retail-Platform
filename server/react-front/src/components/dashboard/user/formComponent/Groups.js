import React, {Component} from "react";
import * as PropTypes from "prop-types";
import Typography from "@material-ui/core/Typography";
import TextField from "@material-ui/core/TextField";
import IconButton from "@material-ui/core/IconButton";
import Icon from "@material-ui/core/Icon";
import Grid from "@material-ui/core/Grid";
import withStyles from "@material-ui/core/styles/withStyles";
import Collapse from "@material-ui/core/Collapse";
import Grow from "@material-ui/core/Grow";

const styles = {
    textField: {}
};

class Groups extends Component {

    state = {groupOn: []};

    componentDidMount() {
        this.groupOn = this.props.groups.map(() => true);
    }

    validation = () => {
        this.props.handleUpdateValidation({error: false});

        return false;
    };

    handleChange = (event, index) => {
        const {groups: _groups, onChange} = this.props;
        const groups = [..._groups];
        groups[index] = event.target.value;
        onChange(groups);
    };

    handleAddGroup = () => {

        const handleAdded = () => {
            this.setState(state => ({groupOn: [...state.groupOn, true]}));
        };

        const {groups, onChange} = this.props;
        onChange([...groups, ""], handleAdded)
    };

    handleOnDrop = index => {
        this.setState(state => {
            const groupOn = state.groupOn.slice();
            groupOn[index] = false;
            return ({groupOn: groupOn});
        })
    };

    handleDropGroup = index => {

        const handleDropped = () => {
            this.setState(state => ({groupOn: state.groupOn.filter(x => x)}));
        };

        const {groups: _groups, onChange} = this.props;
        _groups.splice(index, 1);
        onChange([..._groups], handleDropped);
    };

    render() {
        const {classes, groups} = this.props;
        const {groupOn} = this.state;

        return (
            <div>
                <Typography variant="subtitle1" color="textSecondary" gutterBottom>Groups</Typography>
                {groups.map((group, index) => (
                    <Collapse key={index} in={groupOn[index]} onExited={() => this.handleDropGroup(index)}>
                        <Grow in={groupOn[index]}>
                            <Grid spacing={8} alignItems="flex-end" container>
                                <Grid item>
                                    <IconButton onClick={() => this.handleOnDrop(index)}>
                                        <Icon className="fas fa-minus-circle" color="error" fontSize="small"/>
                                    </IconButton>
                                </Grid>
                                <Grid>
                                    <TextField key={index}
                                               id={`group-${index}`}
                                               className={classes.textField}
                                               value={group}
                                               onChange={e => this.handleChange(e, index)}
                                               onBlur={this.validation}
                                               margin="dense"
                                               fullWidth/>
                                </Grid>
                            </Grid>
                        </Grow>
                    </Collapse>
                ))}
                <Grid spacing={8} alignItems="center" justify="flex-start" container>
                    <Grid item>
                        <IconButton onClick={this.handleAddGroup}>
                            <Icon className="fas fa-plus-circle" color="secondary" fontSize="small"/>
                        </IconButton>
                    </Grid>
                    <Grid item>
                        <Typography align="center">Add new group</Typography>
                    </Grid>
                </Grid>
            </div>)
    }
}

Groups.propTypes = {
    classes: PropTypes.object.isRequired,
    groups: PropTypes.arrayOf(PropTypes.string.isRequired).isRequired,
    handleUpdateValidation: PropTypes.func.isRequired,
};

export default withStyles(styles)(Groups);
