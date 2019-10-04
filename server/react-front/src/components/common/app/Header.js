import React from 'react';
import * as PropTypes from 'prop-types';
import AppBar from '@material-ui/core/AppBar'
import Toolbar from "@material-ui/core/Toolbar";
import SearchIcon from "@material-ui/icons/Search"
import {withStyles} from "@material-ui/core/styles";
import InputBase from "@material-ui/core/InputBase";
import {fade} from "@material-ui/core/styles/colorManipulator";
import Link from "../Link";
import Grid from "@material-ui/core/Grid/Grid";
import classNames from "classnames";
import IconButton from "@material-ui/core/IconButton";
import VideoCallIcon from "@material-ui/icons/VideoCallRounded";
import Login from "./Login";

const styles = theme => ({
    root: {
        flexGrow: 1,
        zIndex: theme.zIndex.drawer + 1,
        width: '100%',
    },
    grow: {
        flexGrow: 1,
    },
    logo: {
        marginLeft: -12,
        marginRight: '8rem',
    },
    imgContainer: {
        margin: 'auto',
        display: 'block',
        maxWidth: '100%',
        maxHeight: '100%',
    },
    img: {
        height: 36,
        '@media (max-width: 840px)': {
            height: 0,
        },
    },
    headerItem: {
        marginRight: 20,
    },
    search: {
        position: 'relative',
        borderRadius: theme.shape.buttonBorderRadius,
        backgroundColor: fade(theme.palette.common.white, 0.15),
        '&:hover': {
            backgroundColor: fade(theme.palette.common.white, 0.25),
        },
        marginLeft: 0,
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            marginLeft: theme.spacing.unit,
            width: 'auto',
        },
    },
    searchIcon: {
        width: theme.spacing.unit * 9,
        height: '100%',
        position: 'absolute',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    inputRoot: {
        color: 'inherit',
        width: '100%',
    },
    inputInput: {
        paddingTop: theme.spacing.unit,
        paddingRight: theme.spacing.unit,
        paddingBottom: theme.spacing.unit,
        paddingLeft: theme.spacing.unit * 10,
        transition: theme.transitions.create('width'),
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            width: 120,
            '&:focus': {
                width: 200,
            },
        },
    },
});

const header = {
    name: ['APIs', 'Hysia Team', 'Github'],
    url: ['/api', '/HysiaTeam', 'https://github.com/HuaizhengZhang/Hysia']
};

const Header = ({classes}) => (
    <AppBar className={classes.root} color={'primary'} position="fixed">
        <Toolbar>
            <Grid item className={classes.imgContainer}>
                <img className={classes.img} src={require('../../../asset/hysia-small.svg')} alt={'hysia-logo-small'}/>
            </Grid>
            <Link className={classes.logo} aira-label={'hysia'} to={'/dashboard'} noWrap>
                Hysia
            </Link>
            {header.name.map((item, key) =>
                <Link key={key} aira-label={item} className={classes.headerItem} to={header.url[key]}
                      variant={'subtitle1'} noWrap>
                    {item}
                </Link>)}
            <div className={classes.grow}/>
            <div className={classNames(classes.headerItem, classes.search)}>
                <div className={classes.searchIcon}>
                    <SearchIcon/>
                </div>
                <InputBase
                    placeholder="Search…"
                    classes={{
                        root: classes.inputRoot,
                        input: classes.inputInput,
                    }}
                />
            </div>
            <Link to={"/dashboard/video/upload"} animation={false}>
                <IconButton color={"inherit"}>
                    <VideoCallIcon/>
                </IconButton>
            </Link>
            <Login/>
        </Toolbar>
    </AppBar>
);

Header.propTypes = {
    classes: PropTypes.object.isRequired
};

export default withStyles(styles)(Header);
